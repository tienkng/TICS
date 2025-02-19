import argparse


def get_args():
    """ Gets command-line arguments.

    Returns:
        Return command-line arguments as a set of attributes.
    """

    parser = argparse.ArgumentParser(description='Train WB Correction.')
    parser.add_argument('--project-name', type=str, default="CWCC_WB",
                        help="Wandb project name logging tensorboard")
    parser.add_argument('--model-config', type=str, default="CWCC_WB", 
                        help="Building model via config yaml")
    
    # Trainer params
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, 
                        help='Batch size')
    parser.add_argument('--optimizer', type=str,
                        default='Adam', help='Adam or SGD')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--device', nargs='+', default=0, type=int)
    parser.add_argument('--accelerator', default='auto', help='cpu, gpu, tpu or auto')
    parser.add_argument('--output-path', default='./output', help='saved logging and dir path')
    parser.add_argument('--log-steps', type=int, default=10,
                        help='Learning rate')
    
    # dataset params
    parser.add_argument('--start-sec', default=0, type=int)
    parser.add_argument('--num-frames', default=32, type=int)
    parser.add_argument('--sampling-rate', default=2, type=int)
    parser.add_argument('--frames-per-second', default=30, type=int)
    parser.add_argument('--do-train', action='store_true', help='Do training')
    parser.add_argument('--training-dir', default='./data/images/', type=str,
                        help='Training directory')
    parser.add_argument('--do-eval', action='store_true', help='Do validation')
    parser.add_argument('--validation-dir', type=str, default=None, 
                        help='Main validation directory')
    parser.add_argument('--num-workers', type=int,
                        default=8, help='Number of workers processing')

    return parser.parse_args()