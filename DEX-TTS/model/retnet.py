from model.retention import *
from model.retnet_cfg import *


class RetNetModel(RetNetPreTrainedModel):

    def __init__(self,
                 config: RetNetConfig,
                 embed_tokens: nn.Embedding = None,
                 tensor_parallel: bool = False,
                 use_softmax: bool = False,
                 use_decay: bool = False):
        super().__init__(config)
        self.config = config

        self.dropout_module = torch.nn.Dropout(config.dropout)

        self.embed_dim = config.decoder_embed_dim
        self.embed_scale = 1.0 if config.no_scale_embedding else math.sqrt(self.embed_dim)

        self.layers = nn.ModuleList([])

        for i in range(config.decoder_layers):
            self.layers.append(RetNetDecoderLayer(config, depth=i, tensor_parallel=tensor_parallel, use_softmax=use_softmax))

        self.decoder_layers = len(self.layers)

        if config.decoder_normalize_before:
            self.layer_norm = RMSNorm(self.embed_dim, eps=config.layernorm_eps)
        else:
            self.layer_norm = None

        self.retnet_rel_pos = RetNetRelPos(config, use_decay=use_decay)
        self.recurrent_chunk_size = config.recurrent_chunk_size

        if config.deepnorm:
            init_scale = math.pow(8.0 * config.decoder_layers, 0.25)
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.div_(init_scale)

        if config.subln and not config.use_glu:
            init_scale = math.sqrt(math.log(config.decoder_layers * 2))
            for name, p in self.named_parameters():
                if ("fc1" in name or "fc2" in name or "out_proj" in name or "v_proj" in name):
                    p.data.mul_(init_scale)

        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        retention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        sty: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Dict[str, torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_retentions: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        use_cache: Optional[bool] = None,
        forward_impl: Optional[str] = 'parallel',
        recurrent_chunk_size: Optional[int] = None,
        retention_rel_pos: Optional[Tuple[torch.Tensor]] = None,
    ) -> Union[Tuple, RetNetOutputWithPast]:

        if output_retentions is None and output_attentions is not None:
            output_retentions = output_attentions
        output_retentions = output_retentions if output_retentions is not None else self.config.output_retentions
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else
                                self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")


        if retention_mask is None and attention_mask is not None:
            retention_mask = attention_mask
        if retention_mask is not None and forward_impl == 'recurrent':
            retention_mask = retention_mask[:, -1:]

        hidden_states = inputs_embeds
        
        # print('mask', retention_mask, attention_mask)

        # handling chunking here
        if recurrent_chunk_size is None:
            recurrent_chunk_size = self.recurrent_chunk_size
        need_pad_for_chunkwise = (forward_impl == 'chunkwise' and
                                  seq_length % recurrent_chunk_size != 0)
        if need_pad_for_chunkwise:
            padding_len = recurrent_chunk_size - seq_length % recurrent_chunk_size
            slen = seq_length + padding_len
            hidden_states = F.pad(hidden_states, (0, 0, 0, padding_len))
        else:
            slen = seq_length
        # relative position
        if retention_rel_pos is None:
            retention_rel_pos = self.retnet_rel_pos(slen,
                                                    forward_impl=forward_impl,
                                                    recurrent_chunk_size=recurrent_chunk_size,
                                                    retention_mask=retention_mask,
                                                    get_decay_scale=not self.training)

        # start running through the decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_retentions = () if output_retentions else None
        # layers * [bsz, num_head, qk_dim, decoder_embed_dim]
        next_decoder_cache = () if use_cache else None

        for idx, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):

                    def custom_forward(*inputs):
                        return module(*inputs, output_retentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    retention_rel_pos,
                    sty,
                    retention_mask,
                    forward_impl,
                    past_key_value,
                )
            else:
                layer_outputs = layer(hidden_states,
                                      retention_rel_pos,
                                      sty,
                                      retention_mask=retention_mask,
                                      forward_impl=forward_impl,
                                      past_key_value=past_key_value,
                                      output_retentions=output_retentions)

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[1],)

            if output_retentions:
                all_retentions += (layer_outputs[2],)

        next_cache = next_decoder_cache if use_cache else None

        if need_pad_for_chunkwise:
            hidden_states = hidden_states[:, :seq_length, :]

        if self.layer_norm is not None:
            hidden_states = self.layer_norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return RetNetOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            retentions=all_retentions,
            attentions=all_retentions,
        )
        
if __name__ == '__main__':

    config = RetNetConfig(decoder_layers=8,
                        decoder_embed_dim=512,
                        decoder_value_embed_dim=1024,
                        decoder_retention_heads=4,
                        decoder_ffn_embed_dim=1024)
    model = RetNetModel(config)

    input_ids = torch.LongTensor([[1,2,3,4,5,6,7,8]])

    parallel_outputs = model(input_ids, forward_impl='parallel', use_cache=True)
    parallel_state = parallel_outputs.last_hidden_state
    parallel_cache = parallel_outputs.past_key_values
    
    print(input_ids.shape)
    print(parallel_outputs[0].shape)