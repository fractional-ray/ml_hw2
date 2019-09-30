"""
Functions for training and predicting with linear classifiers
"""
import numpy as np
import pylab as plt
from scipy.optimize import minimize, check_grad


def linear_predict(data, model):
    """
    Predicts a multi-class output based on scores from linear combinations of features. 
    
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :return: length n vector of class predictions
    :rtype: array
    """
    
    value = model["weights"].T.dot(data)
    return value.argmax(0)
        
    

def perceptron_update(data, model, label):
    """
    Update the model based on the perceptron update rules and return whether the perceptron was correct
    
    :param data: (d, 1) ndarray representing one example input
    :type data: ndarray
    :param model: dictionary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param label: the class label of the single example
    :type label: int
    :return: whether the perceptron correctly predicted the provided true label of the example
    :rtype: bool
    """
    f_x = linear_predict(data,model)
    if f_x == label:
        return True 
    else:
        model["weights"][:,f_x] -=  data 
        model["weights"][:,label] +=  data
        return False
    


def log_reg_train(data, labels, params, model=None, check_gradient=False):
    """
    Train a linear classifier by maximizing the logistic likelihood (minimizing the negative log logistic likelihood)
     
    :param data: size (d, n) ndarray containing n examples described by d features each
    :type data: ndarray
    :param labels: length n array of the integer class labels 
    :type labels: array
    :param params: dictionary containing 'lambda' key. Lambda is the regularization parameter and it should be a float
    :type params: dict
    :param model: dictio`nary containing 'weights' key. The value for the 'weights' key is a size 
                    (d, num_classes) ndarray
    :type model: dict
    :param check_gradient: Boolean value indicating whether to run the numerical gradient check, which will skip
                            learning after checking the gradient on the initial model weights.
    :type check_gradient: Boolean
    :return: the learned model 
    :rtype: dict
    """
    d, n = data.shape
    num_classes = np.unique(labels).size

    if model:
        weights = model['weights'].ravel()
    else:
        weights = np.zeros(d * num_classes)

    def log_reg_nll(new_weights):
        """
        This internal function returns the negative log-likelihood (nll) of the data given the logistic regression weights
        
        :param new_weights: weights to use for computing logistic regression likelihood
        :type new_weights: ndarray
        :return: tuple containing (<negative log likelihood of data>, gradient)
        :rtype: tuple
        """
        # reshape the weights, which the optimizer prefers to be a vector, to the more convenient matrix form
        new_weights = new_weights.reshape((d, num_classes))
        c_lambda = params["lambda"]
        nll_gradient = tuple()
        gradient = np.zeros(shape = (d,num_classes))
        inner_term, last_term = 0, 0
        #print(new_weights)
        #print(model)

        frob_norm = np.linalg.norm(new_weights,ord="fro")
        first_term = c_lambda / 2.0 * frob_norm ** 2 
        # TODO fill in your code here to compute the objective value (nll)
        for index in range(n):
            #for c in range(d):
                #inner_term += np.log(np.exp(new_weights[:,c].T.dot(data[i])))
            #inner_term += np.log(np.exp(new_weights.T.dot(data[:,index])))
            y_i = labels[index]
            inner_term += logsumexp(new_weights.T.dot(data[:,index]),dim = 0)
            last_term += new_weights[:,y_i].T.dot(data[:,index])



        nnl = first_term + inner_term - last_term

        # TODO fill in your code here to compute the gradient
        #for i in range(n):
        #    y_i = labels[i]
        #    for c in range(d):
        
        for c in range(num_classes):
            first_term = c_lambda * new_weights[:,c] 
            for i in range(n):
                y_i = labels[i]
                s = new_weights.T.dot(data[:,i])
                scalar = new_weights[:,c].T.dot(data[:,i]) - logsumexp(s, dim = 0)
                scalar = np.exp(scalar)
                if y_i == c:
                    #last_term = new_weights[:,y_i].T.dot(data[:,i])
                    last_term = 1
                    #gradient[:,c] = first_term + data[:, i] * scalar + 5 *  last_term

                else:
                    #last_term = new_weights[:,y_i].T.dot(data[:,i])
                    last_term = 0 
                    #gradient[:,c] = first_term + data[:, i] * scalar - 5 *  last_term 
                gradient[:,c] += data[:, i] * (scalar - last_term)
            gradient[:,c] += first_term 
      

        #print(gradient.shape)

        nll_gradient = (nnl,gradient)
        return nll_gradient 
        #return nll, gradient

    if check_gradient:
        grad_error = check_grad(lambda w: log_reg_nll(w)[0], lambda w: log_reg_nll(w)[1].ravel(), weights)
        print("Provided gradient differed from numerical approximation by %e (should be around 1e-3 or less)" % grad_error)
        return model

    # pass the internal objective function into the optimizer
    res = minimize(lambda w: log_reg_nll(w)[0], jac=lambda w: log_reg_nll(w)[1].ravel(), x0=weights)
    weights = res.x
    model = {'weights': weights.reshape((d, num_classes))}

    return model


def plot_predictions(data, labels, predictions):
    """
    Utility function to visualize 2d, 4-class data 
    
    :param data: 
    :type data: 
    :param labels: 
    :type labels: 
    :param predictions: 
    :type predictions: 
    :return: 
    :rtype: 
    """
    num_classes = np.unique(labels).size

    markers = ['x', 'o', '*',  'd']

    for i in range(num_classes):
        plt.plot(data[0, np.logical_and(labels == i, labels == predictions)],
                 data[1, np.logical_and(labels == i, labels == predictions)],
                 markers[i] + 'g')
        plt.plot(data[0, np.logical_and(labels == i, labels != predictions)],
                 data[1, np.logical_and(labels == i, labels != predictions)],
                 markers[i] + 'r')


def logsumexp(matrix, dim=None):
    """
    Compute log(sum(exp(matrix), dim)) in a numerically stable way.
    
    :param matrix: input ndarray
    :type matrix: ndarray
    :param dim: integer indicating which dimension to sum along
    :type dim: int
    :return: numerically stable equivalent of np.log(np.sum(np.exp(matrix), dim)))
    :rtype: ndarray
    """
    try:
        with np.errstate(over='raise', under='raise'):
            return np.log(np.sum(np.exp(matrix), dim, keepdims=True))
    except:
        max_val = np.nan_to_num(matrix.max(axis=dim, keepdims=True))
        with np.errstate(under='ignore', divide='ignore'):
            return np.log(np.sum(np.exp(matrix - max_val), dim, keepdims=True)) + max_val
