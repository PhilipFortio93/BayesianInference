# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

def get_weights(x, y, verbose = 0):
    shape = x.shape
    print('x shape = ', shape)
    x = np.insert(x, 0, 1, axis=1)
    w = np.ones((shape[1]+1,))
    weights = []
    
    learning_rate = 10.0
    iteration = 0
    loss = 100
    while iteration <= 1000 and loss > 0.1:
        pred_conf = []
        for ix, i in enumerate(x):

            pred = np.dot(i,w)/100

            try:
                pos_conf=np.exp(pred)
                neg_conf=np.exp(pred*-1)
            except:
                if(pred>0):
                    pos_conf = 1
                    neg_conf = 0
                else:
                    pos_conf = 0
                    neg_conf = 1
            norm = pos_conf+neg_conf
            pos_conf = pos_conf/norm
            neg_conf = neg_conf/norm

            if pred > 0: 
                pred = pos_conf
                pred_conf.append(pos_conf)
            elif pred < 0: 
                pred = -1*neg_conf
                pred_conf.append(neg_conf)
            # if pred != y[ix]:
            w = w - learning_rate * (pred-y[ix]) * i
            weights.append(w)    
            if verbose == 1:
                print('X_i = ', i, '    y = ', y[ix])
                print('Pred: ', pred )
                print('Weights', w)
                print('------------------------------------------')
        print('w',w)
        print('rate: ', learning_rate)
        loss = np.dot(x, w)
            
        
        loss[loss<0] = -1
        loss[loss>0] = 1
        loss = np.multiply(loss,pred_conf)
        print(loss)
        dif = abs((loss-y))
        print('dif: ',dif)
        loss = np.sum(dif)

        if verbose == 1:
            print('------------------------------------------')
            print(loss)
            print('------------------------------------------')
        if iteration%100 == 0: learning_rate = learning_rate / 2.0
        iteration += 1    
         
    print('Weights: ', w)
    print('Loss: ', loss)
    return w, weights

data = np.array([[3,4,1],[2,3,1],[5,1,1],[10,9,-1],[11,11,-1],[1,10,1],[10,6,-1],[7,3,-1]])

x = data[:,:-1]
y = data[:,-1]

print('Dataset')
print('x = ',x)
print('y = ', y)

w, all_weights = get_weights(x, y)
x = np.insert(x, 0, 1, axis=1)

pred = np.dot(x, w)
pred[pred > 0] =  1
pred[pred < 0] = -1
print('Predictions', pred)

x1 = np.linspace(np.amin(x[:,1]),np.amax(x[:,2]),2)
x2 = np.zeros((2,))
for ix, i in enumerate(x1):
    x2[ix] = (-w[0] - w[1]*i) / w[2]

plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
plt.plot(x1,x2)
plt.title('Perceptron Seperator', fontsize=20)
plt.xlabel('Feature 1 ($x_1$)', fontsize=16)
plt.ylabel('Feature 2 ($x_2$)', fontsize=16)
plt.show()


for ix, w in enumerate(all_weights):
    if ix % 500 == 0:
        print ('iteration: ',ix/10)
        print('Weights:', w)
        x1 = np.linspace(np.amin(x[:,1]),np.amax(x[:,2]),2)
        x2 = np.zeros((2,))
        for ix, i in enumerate(x1):
            x2[ix] = (-w[0] - w[1]*i) / w[2]
        print('$0 = ' + str(-w[0]) + ' - ' + str(w[1]) + 'x_1'+ ' - ' + str(w[2]) + 'x_2$')

        plt.scatter(x[y>0][:,1], x[y>0][:,2], marker = 'x')
        plt.scatter(x[y<0][:,1], x[y<0][:,2], marker = 'o')
        plt.plot(x1,x2)
        plt.title('Perceptron Seperator', fontsize=20)
        plt.xlabel('Feature 1 ($x_1$)', fontsize=16)
        plt.ylabel('Feature 2 ($x_2$)', fontsize=16)
        plt.show()
