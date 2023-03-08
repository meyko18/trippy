class TransformerForDST(PARENT_CLASSES[parent_name]):
        def __init__(self, config):
            assert config.model_type in PARENT_CLASSES
            assert self.__class__.__bases__[0] in MODEL_CLASSES
            super(TransformerForDST, self).__init__(config)
            self.model_type = config.model_type
            self.slot_list = config.dst_slot_list
            self.class_types = config.dst_class_types
            self.class_labels = config.dst_class_labels
            self.token_loss_for_nonpointable = config.dst_token_loss_for_nonpointable
            self.refer_loss_for_nonpointable = config.dst_refer_loss_for_nonpointable
            self.class_aux_feats_inform = config.dst_class_aux_feats_inform
            self.class_aux_feats_ds = config.dst_class_aux_feats_ds
            self.class_loss_ratio = config.dst_class_loss_ratio

            # Only use refer loss if refer class is present in dataset.
            if 'refer' in self.class_types:
                self.refer_index = self.class_types.index('refer')
            else:
                self.refer_index = -1

            # Make sure this module has the same name as in the pretrained checkpoint you want to load!
            self.add_module(self.model_type, MODEL_CLASSES[self.__class__.__bases__[0]](config))
            if self.model_type == "electra":
                self.pooler = ElectraPooler(config)
            
            self.dropout = nn.Dropout(config.dst_dropout_rate)
            self.dropout_heads = nn.Dropout(config.dst_heads_dropout_rate)

            if self.class_aux_feats_inform:
                self.add_module("inform_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))
            if self.class_aux_feats_ds:
                self.add_module("ds_projection", nn.Linear(len(self.slot_list), len(self.slot_list)))

            aux_dims = len(self.slot_list) * (self.class_aux_feats_inform + self.class_aux_feats_ds) # second term is 0, 1 or 2

            for slot in self.slot_list:
                self.add_module("class_" + slot, nn.Linear(config.hidden_size + aux_dims, self.class_labels))
                self.add_module("token_" + slot, nn.Linear(config.hidden_size, 2))
                self.add_module("refer_" + slot, nn.Linear(config.hidden_size + aux_dims, len(self.slot_list) + 1))

            # Head for aux task
            if hasattr(config, "aux_task_def") and config.aux_task_def is not None:
                self.add_module("aux_out_projection", nn.Linear(config.hidden_size, int(config.aux_task_def['n_class'])))

            self.init_weights()