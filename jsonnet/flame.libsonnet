{
  normLr(lr, batch_size, base_batch_size=256):: (
    // normalize lr后面应该会放到python里面做
    lr * batch_size / base_batch_size
  ),
  DataModule(train_loader=null, val_loader=null, test_loader=null):: {
    _call: 'flame.experimental.trainer.DataModule',
    train_loader: train_loader,
    val_loader: val_loader,
    test_loader: test_loader,
  },
}
