import random

def next_seed(args, root):
    if args.seed_behavior == 'iter':
        root.seed_internal += 1
        if root.seed_internal >= args.seed_iter_N:
            root.seed_internal = 0
            args.seed += 1
    elif args.seed_behavior == 'ladder':
        args.seed += 2 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'alternate':
        args.seed += 1 if root.seed_internal == 0 else -1
        root.seed_internal = 1 if root.seed_internal == 0 else 0
    elif args.seed_behavior == 'fixed':
        pass  # always keep seed the same
    else:
        args.seed = random.randint(0, 2**32 - 1)
    return args.seed
