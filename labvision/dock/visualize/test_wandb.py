# Inside my model training code
import wandb
wandb.init(project="my-project")

wandb.config.dropout = 0.2
wandb.config.hidden_layer_size = 128

# def my_train_loop():
#     for epoch in range(10):
#         loss = 0  # change as appropriate :)
#         wandb.log({'epoch': epoch, 'loss': loss})


# my_train_loop()
