# Include some training method
from torch.utils.data import Dataset, DataLoader

from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV


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


def GridSearch(model, Net_parameters, Search_parameters, dataset):
    '''
    Do the grid search with skorch and sklearn
    '''
    net = NeuralNetRegressor(**Net_parameters )

    gs = GridSearchCV(net, ** Search_parameters)

    x , y, z = dataset.data.shape

    data = dataset.data.reshape((x, y*z))

    x , y, z = dataset.labels.shape

    labels = dataset.labels.reshape((x, y*z))

    gs.fit(data, labels)

    print("best score: {:.3f}".format(gs.best_score_))

    return gs.best_estimator_
