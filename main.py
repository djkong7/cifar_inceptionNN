import pickle, os, sys, time
import torch as pt, numpy as np
import matplotlib.pyplot as plt

LEARNING_RATE = .001
NUM_CLASSES = 10
BATCHSIZE = 100
OUTPUT_SIZE = 10

class Net(pt.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        #ouput size = (input_dims-kernel_size)/stride + 1
        #(36-5)/1+1 = 32
        self.conv_5x5_a = pt.nn.Conv2d(in_channels = 3, out_channels = 14, kernel_size = 5, padding = 2)
        self.conv_3x3_a = pt.nn.Conv2d(in_channels = 3, out_channels = 15, kernel_size = 3, padding = 1)
        self.pool_3x3_a = pt.nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        
        #(20-5)/1+1 = 16
        self.conv_5x5_b = pt.nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 5, padding = 2)
        self.conv_3x3_b = pt.nn.Conv2d(in_channels = 32, out_channels = 16, kernel_size = 3, padding = 1)
        self.pool_3x3_b = pt.nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1)
        
        self.half_pool = pt.nn.AvgPool2d(kernel_size = 2)
        self.quarter_pool = pt.nn.AvgPool2d(kernel_size = 4)
        
        self.fc1 = pt.nn.Linear(1024,256)
        self.fc2 = pt.nn.Linear(256,10)
        
    def forward(self, x):
        d = x.clone()
        e = x.clone()
        f = x.clone()
        #inception module
        d = pt.nn.functional.relu(self.conv_5x5_a(d))
        e = pt.nn.functional.relu(self.conv_3x3_a(e))
        f = pt.nn.functional.relu(self.pool_3x3_a(f))
        x = pt.cat((d,e,f),1)
        
        x = self.half_pool(x)
        
        d = x.clone()
        e = x.clone()
        f = x.clone()
        #inception module
        d = pt.nn.functional.relu(self.conv_5x5_b(d))
        e = pt.nn.functional.relu(self.conv_3x3_b(e))
        f = pt.nn.functional.relu(self.pool_3x3_b(f))
        x = pt.cat((d,e,f),1)
        
        x = self.quarter_pool(x)
        
        x = x.view(-1,1024)
        #print(x.shape)
        x = self.fc1(x)
        x = pt.nn.functional.relu(x)
        x = self.fc2(x)
        return x
    
    import subprocess

    
        

def train(net, train_images, train_labels, validate_images, validate_labels):
    num_epoch = 1000
    stop = False
 
    loss = np.Inf
    counter = 0
    loss_increase = 0

    #For calculating loss
    criterion = pt.nn.modules.loss.CrossEntropyLoss()
    optimizer = pt.optim.SGD(net.parameters(), lr = LEARNING_RATE, momentum = 0.9)
    
    if pt.cuda.is_available():
        criterion = criterion.cuda()
    
    for j in range(0,num_epoch):
        print("Epoch number", j)
        start = time.time()#Just for timing
        for i in range(0,train_images.shape[0], BATCHSIZE):
        #for i in range(0,1,nn.BATCHSIZE):
            #create matricies of training data and labels
            data = train_images[i:i+BATCHSIZE]
            labels = train_labels[i:i+BATCHSIZE]

            # wrap them in Variable
            (data, labels) = pt.autograd.Variable(data), pt.autograd.Variable(labels)
            
            #zero the parameter gradients
            optimizer.zero_grad()
            y = net(data)
            new_loss = criterion(y,labels)
            new_loss.backward()
            optimizer.step()

            counter += 1

            if (counter % 150) == 0:

                #Automatically stop to prevent overtraining.
                if (new_loss.data[0] - loss) <= 0:
                    loss = new_loss.data[0]
                    loss_increase = 0
                    pt.save(net.state_dict(),"model_save/model")
                    #print("Saved!")
                    print("\tLoss: %f"%(new_loss.data[0]))
                    #test(validate_images,validate_labels)
                else:
                    loss_increase += 1
                    print(loss_increase)
                    if loss_increase >= 8:
                        #Stop epoch loop
                        stop = True
                        #Stop batchsize loop
                        break

        end = time.time()
        print("\tTime for Epoch:",end-start)
        if stop:
            break

def test(test_images, test_labels):
    test_images = pt.autograd.Variable(test_images)
    y = net(test_images)
    _, predicted = pt.max(y.data,1)
    correct = (predicted == test_labels).sum()
    
    print("\tAccuracy: %.2f%%" %(correct/test_labels.shape[0]*100))

def load_data():

    if not os.path.exists("model_save"):
                os.makedirs("model_save")
                    

    def unpickle(file):
        with open(file, 'rb') as fo:
            diction = pickle.load(fo, encoding='bytes')
        return diction

    try:
        print("Loading pytorch data.")
        #Pytorch files are much faster to load than the pickled data; about 6X faster
        #Try to load these first
        train_images = pt.load("data/pytorch/train_images.pt")
        train_labels = pt.load("data/pytorch/train_labels.pt")
        test_images = pt.load("data/pytorch/test_images.pt")
        test_labels = pt.load("data/pytorch/test_labels.pt")

    except IOError as e1:
        print(e1)
        print("Pytorch hasn't been written. Trying pickled...")

        try:
            #Load the data from pytorch
            np_train_data_1 = unpickle("data/cifar-10-batches-py/data_batch_1")
            np_train_data_2 = unpickle("data/cifar-10-batches-py/data_batch_2")
            np_train_data_3 = unpickle("data/cifar-10-batches-py/data_batch_3")
            np_train_data_4 = unpickle("data/cifar-10-batches-py/data_batch_4")
            np_train_data_5 = unpickle("data/cifar-10-batches-py/data_batch_5")
            np_test_data = unpickle("data/cifar-10-batches-py/test_batch")
            
            
            #np to pt, view to a 3d image with 3 channels, normalize to between 0 and 1
            train_images_1 = pt.from_numpy(np_train_data_1[b"data"]).float().view(-1,3,32,32)/255
            train_images_2 = pt.from_numpy(np_train_data_2[b"data"]).float().view(-1,3,32,32)/255
            train_images_3 = pt.from_numpy(np_train_data_3[b"data"]).float().view(-1,3,32,32)/255
            train_images_4 = pt.from_numpy(np_train_data_4[b"data"]).float().view(-1,3,32,32)/255
            train_images_5 = pt.from_numpy(np_train_data_5[b"data"]).float().view(-1,3,32,32)/255
            test_images = pt.from_numpy(np_test_data[b"data"]).float().view(-1,3,32,32)/255
            
            #extract labels
            train_labels_1 = pt.LongTensor(np_train_data_1[b"labels"])
            train_labels_2 = pt.LongTensor(np_train_data_2[b"labels"])
            train_labels_3 = pt.LongTensor(np_train_data_3[b"labels"])
            train_labels_4 = pt.LongTensor(np_train_data_4[b"labels"])
            train_labels_5 = pt.LongTensor(np_train_data_5[b"labels"])
            test_labels = pt.LongTensor(np_test_data[b"labels"])
            
            train_images = pt.cat((train_images_1, train_images_2, train_images_3, train_images_4, train_images_5),0)
            train_labels = pt.cat((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5),0)
            #Verify proper concatenation
            #print(train_images.shape)
            #print(train_labels.shape)

            #Make the pytorch directory if itwmu_test_images, wmu_test_labels doesn't exist.
            if not os.path.exists("data/pytorch"):
                os.makedirs("data/pytorch")

            pt.save(train_images, "data/pytorch/train_images.pt")
            pt.save(train_labels, "data/pytorch/train_labels.pt")
            pt.save(test_images, "data/pytorch/test_images.pt")
            pt.save(test_labels, "data/pytorch/test_labels.pt")
            
        except IOError as e2:
            print("Couldn't find file.", e2)
            print("Exiting...")
            sys.exit(1)
    
    try:
        #no easy way to save these. A little unpickle never hurt anyone ;)
        np_batches_meta = unpickle("data/cifar-10-batches-py/batches.meta")
        label_labels = np_batches_meta[b"label_names"]
        #print(label_labels[0])
    except IOError as e:
        print("Couldn't find file.", e)
        print("Exiting...")
        sys.exit(1)
        
    #View an image from the dataset
    '''
    num = 30001
    npimg = train_images[num].numpy()
    #Transpose bc it's stored as a 3x32x32 when imgshow needs 32x32x3
    plt.imshow(np.transpose(npimg, (1,2,0)))
    print(label_labels[train_labels[num]])
    '''
    
    #Create validation set
    validate_images = train_images[45000:]
    train_images = train_images[0:45000]
    
    validate_labels = train_labels[45000:]
    train_labels = train_labels[0:45000]

    return train_images, train_labels, validate_images, validate_labels, test_images, test_labels, label_labels

if __name__ == "__main__":

    net = Net()
    #load_start = time.time()
    train_images, train_labels, validate_images, validate_labels, test_images, test_labels, label_labels = load_data()
    #load_end = time.time()
    #print("Time to load", load_end-load_start)
    
    if pt.cuda.is_available():
        print("On Cuda")
        net.cuda()
        train_images = train_images.cuda()
        train_labels = train_labels.cuda()
        validate_images = validate_images.cuda()
        validate_labels = validate_labels.cuda()
        test_images = test_images.cuda()
        test_labels = test_labels.cuda()

    to_train = 1
    continue_train = 0
    if to_train:
        if continue_train:
            if pt.cuda.is_available():
                net.load_state_dict(pt.load("model_save/model"))
            else:
                net.load_state_dict(pt.load("model_save/model", map_location=lambda storage, loc: storage))
            print("loaded")

        train_start = time.time()
        train(net, train_images, train_labels, validate_images, validate_labels)
        train_end = time.time()
        print("Total time to train:", train_end-train_start)

    if pt.cuda.is_available():
        net.load_state_dict(pt.load("model_save/model"))
    else:
        net.load_state_dict(pt.load("model_save/model", map_location=lambda storage, loc: storage))
    
    #full test set uses over 4G vram. nogo on a GTX980
    small_test_images = test_images[0:3500]
    small_test_labels = test_labels[0:3500]


    test(small_test_images, small_test_labels)
