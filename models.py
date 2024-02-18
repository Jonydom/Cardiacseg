from networks.cardiacseg import CardiacSeg


def choose_model(args):
    
    hidden_size, num_heads = vit_params(args)
    
    if args.model_name == 'cardiacseg':
        model = CardiacSeg(
                            in_channels = args.in_channels,
                            out_channels = args.num_classes,
                            img_size = args.image_size,
                            feature_size = 16,    ####
                            hidden_size = hidden_size,
                            mlp_dim = 3072,
                            num_heads = num_heads,
                            norm_name = args.norm,
                            res_block = True,
                            dropout_rate = 0.0,
                            lora = args.lora,
                            res_adpter = args.res_adpter,
                            adapterformer = args.adapterformer,
                            args = args
                        )
    return model


def vit_params(args):
    
    if args.arch == 'vit_tiny':
        hidden_size = 192
        num_heads = 3
    elif args.arch == 'vit_base':
        hidden_size = 768
        num_heads = 12
    elif args.arch == 'vit_large':
        hidden_size = 1152
        num_heads = 16
    elif args.arch == 'vit_huge':
        hidden_size = 1344
        num_heads = 16

    return hidden_size, num_heads