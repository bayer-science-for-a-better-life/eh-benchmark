from eh_benchmark.train import parse_args, train


def test_integration():
    command_line_args = [
        '--max_epochs=1',
        '--limit_train_batches=1',
        '--eh_feat',
        '--n_conv_layers=1',
        '--embedding_dim=32',
        '--dataset_name=freesolv',
    ]
    args = parse_args(command_line_args)
    train(args)
