# Include some training method
from torch.utils.data import Dataset, DataLoader


def train(model, training_data, epoch_num ,batch_size, optimizer, loss_fn, print_out_frequency):
    '''
    Simplest trainning function, take in a model, train it with corrsponding hyper parameters , loss and optimizer. 
    Return the final trained model and loss value
    '''

    train_dataloader = DataLoader(training_data, batch_size = batch_size)


    for epoch in range(epoch_num):
        for batch, (X, y) in enumerate(train_dataloader):
            model.zero_grad()

            # print(X.dtype)
            pred = model(X)
            loss = loss_fn(pred, y)
        
            # backpropagation
        
            loss.backward()
            optimizer.step()
        
            loss = loss.item()

        if epoch % print_out_frequency == 0:
            print("The loss is {} in epoch {}".format(loss ,epoch))


    print(f'Training is finished, the final loss is {loss}')

    return model, loss