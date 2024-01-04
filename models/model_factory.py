

def model_factory(args, num_classes):
    if args.arch.lower() == 'vmm':
        from models.VMM import VMM
        return VMM(args.app, args.basenet, args.size, num_classes=num_classes, normal_size=args.normal_size,
                   fpn_type=args.use_FPN, head_type=args.head_type, cls_head=args.cls_head,
                   activation=args.activation)
    elif args.arch.lower() == 'efficientdet':
        from .EfficientDet import EfficientDet
        return EfficientDet(args.app, args.basenet, args.size, num_classes=num_classes, fpn_type=args.use_FPN,
                            head_type=args.head_type, activation=args.activation,
                            cls_head=args.cls_head, normal_size=args.normal_size, from_scratch=args.from_scratch)
    elif 'deeplab' in args.arch.lower():
        assert all([app in ['segmentation', 'depth', 'normal'] for app in args.app]), \
            'DeepLab architecture only convers pixel-wise prediction tasks'
        from models.DeepLab import DeepLab
        return DeepLab(args.size, num_classes, args.basenet, args.skip_connection, args.shared_head,
                       args.normal_size, args.output_stride, from_scratch=args.from_scratch, applications=args.app)
    else:
        raise NotImplementedError
