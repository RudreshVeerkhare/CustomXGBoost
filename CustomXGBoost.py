import numpy as np
import pandas as pd
class Node:
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1):
        self.x = x
        self.y = y
        self.grad = grad
        self.hess = hess
        self.depth = depth
        self.gamma = gamma
        self.lambda_ = lambda_
        self.min_child_weight = min_child_weight
        self.colsample = colsample
        self.cols = np.random.permutation(x.shape[1])[:round(colsample * x.shape[1])]
        self.sim_score = self.similarity_score([True]*x.shape[0])
        self.gain = float("-inf")
        
        self.split_col = None
        self.split_row = None
        self.lhs_tree = None
        self.rhs_tree = None
        self.pivot = None
        self.val = None
        # making split
        self.split_node()
        
        if self.is_leaf:
            self.val = - np.sum(grad) / (np.sum(hess) + lambda_)
        
    
    def split_node(self):
        
        self.find_split()
        
        # checking whether it's a leaf or not
        if self.is_leaf:
            return
        
        x = self.x[:, self.split_col]
        
        lhs = x <= x[self.split_row]
        rhs = x > x[self.split_row]
        
        # creating further nodes recursivly
        self.lhs_tree = Node(
            self.x[lhs],
            self.y[lhs],
            self.grad[lhs],
            self.hess[lhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
        
        self.rhs_tree = Node(
            self.x[rhs],
            self.y[rhs],
            self.grad[rhs],
            self.hess[rhs],
            depth = self.depth - 1,
            gamma = self.gamma,
            min_child_weight = self.min_child_weight,
            lambda_ = self.lambda_,
            colsample = self.colsample
        )
        
    def find_split(self):
        # iterate through every feature and row
        for c in self.cols:
            x = self.x[:, c]
            for row in range(self.x.shape[0]):
                pivot= x[row]
                lhs = x <= pivot
                rhs = x > pivot
                sim_lhs = self.similarity_score(lhs)
                sim_rhs = self.similarity_score(rhs)
                gain = sim_lhs + sim_rhs - self.sim_score - self.gamma
                
                if gain < 0 or self.not_valid_split(lhs) or self.not_valid_split(rhs):
                    continue
                
                if gain > self.gain:
                    self.split_col = c
                    self.split_row = row
                    self.pivot = pivot
                    self.gain = gain
                    
    def not_valid_split(self, masks):
        if np.sum(self.hess[masks]) < self.min_child_weight:
            return True
        return False
    
    @property
    def is_leaf(self):
        if self.depth < 0 or self.gain == float("-inf"):
            return True
        return False
                
    def similarity_score(self, masks):
        return  np.sum(self.grad[masks]) ** 2 / ( np.sum(self.hess[masks]) + self.lambda_ )
    
    
    def predict(self, x):
        return np.array([self.predict_single_val(row) for row in x])
    
    def predict_single_val(self, x):
        if self.is_leaf:
            return self.val
        
        return self.lhs_tree.predict_single_val(x) if x[self.split_col] <= self.pivot else self.rhs_tree.predict_single_val(x)
    

class XGBTree:
    def __init__(self, x, y, grad, hess, depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        indices = np.random.permutation(x.shape[0])[:round(subsample * x.shape[0])]
        
        self.tree = Node(
            x[indices],
            y[indices],
            grad[indices],
            hess[indices],
            depth = depth,
            gamma = gamma,
            min_child_weight = min_child_weight,
            lambda_ =  lambda_,
            colsample = colsample,
        )
    
    def predict(self, x):
        return self.tree.predict(x)
    


class XGBRegressor:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
        
        
    
    def fit(self, x, y, eval_set = None):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)
            
            if eval_set:
                X = eval_set[0]
                Y = eval_set[1]
                cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
                self.history["test"].append(cost)
                print(f"[{n}] validation_set-rmse : {cost}", end="\t")
            
            cost = np.sqrt(np.mean(self.loss(y, base_pred)))
            self.history["train"].append(cost)
            print(f"[{n}] train_set-rmse : {cost}")
            
    def predict(self, x):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        
        return base_pred
    
    def loss(self, y, a):
        return (y - a)**2
    
    def grad(self, y, a):
        # for 0.5 * (y - a)**2
        return a - y
    
    def hess(self, y, a):
        # for 0.5 * (y - a)**2
        return np.full((y.shape), 1)
    

    
class XGBClassifierBase:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
    
    def fit(self, x, y):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            base_pred = base_pred + self.eta * estimator.predict(x)
            self.trees.append(estimator)
            
            
    def predict(self, x, prob=True):
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        for tree in self.trees:
            base_pred += self.eta * tree.predict(x)
        pred_prob = self.sigmoid(base_pred)
        if prob: return pred_prob
        return np.where(pred_prob > 0.5, 1, 0)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y, a):
        return - (y * np.log(a) + (1 - y) * np.log(1 - a))
    
    def grad(self, y, a):
        a_prob = self.sigmoid(a)
        return a_prob - y
    
    def hess(self, y, a):
        a_prob = self.sigmoid(a)
        return a_prob * (1 - a_prob)
    

    
class XGBClassifier:
    def __init__(self, n_classes, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.n_classes = n_classes
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # list of all binary classifiers learners
        self.trees = list()
        
        for n in range(n_classes):
            tree = XGBClassifierBase(
                eta = eta,
                n_estimators = n_estimators,
                max_depth = max_depth,
                gamma = gamma,
                min_child_weight = min_child_weight,
                lambda_ = lambda_,
                colsample = colsample,
                subsample = subsample
            )
            self.trees.append(tree)
    
    def fit(self, x, y, eval_set = None):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        one_hot_y = self.get_one_hot(y, self.n_classes)
        for n in range(self.n_classes):
            print(f"tree{n+1}")
            y = one_hot_y[:, n]
            tree = self.trees[n]
            tree.fit(x, y)
            
            
    def predict(self, x):
        y = self.trees[0].predict(x).reshape(-1, 1)
        
        for i in range(1, self.n_classes):
            y = np.concatenate((y, self.trees[i].predict(x).reshape(-1, 1)), axis = 1)
        
        return y.argmax(axis=1)
    
    def loss(self, y, a):
        return (y - a)**2
    
    @staticmethod
    def get_one_hot(target, nb_classes):
        one_hot = np.zeros((target.shape[0], nb_classes))
        rows = np.arange(target.shape[0])
        one_hot[rows, target] = 1
        return one_hot

    
class XGBRegressorAdam:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
        
        # adam params
        self.b1 = 0.9
        self.b2 =0.999
        self.epsilon = 1e-7
        
        
    
    def fit(self, x, y, eval_set = None):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        vd = np.full(y.shape, 0).astype("float64")
        sd = np.full(y.shape, 0).astype("float64")
        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            dw = estimator.predict(x)
            #####
            vd = self.b1 * vd + (1 - self.b1) * dw
            sd = self.b2 * sd + (1 - self.b2) * (dw**2)
            lr = self.eta #np.sqrt(1 - self.b2**(n+1)) / (1 - self.b1**(n+1))
            #####
            base_pred = base_pred +  lr * vd / (np.sqrt(sd) + self.epsilon)
            self.trees.append(estimator)
            
            if eval_set:
                X = eval_set[0]
                Y = eval_set[1]
                cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
                self.history["test"].append(cost)
                print(f"[{n}] validation_set-rmse : {cost}", end="\t")
            
            cost = np.sqrt(np.mean(self.loss(y, base_pred)))
            self.history["train"].append(cost)
            print(f"[{n}] train_set-rmse : {cost}")
            
    def predict(self, x):
        vd = np.full((x.shape[0],), 0).astype("float64")
        sd = np.full((x.shape[0],), 0).astype("float64")
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        n = 1
        for tree in self.trees:
            dw = tree.predict(x)
            #####
            vd = self.b1 * vd + (1 - self.b1) * dw
            sd = self.b2 * sd + (1 - self.b2) * (dw**2)
            lr = self.eta #np.sqrt(1 - self.b2**(n+1)) / (1 - self.b1**(n+1))
            n += 1
            #####
            base_pred +=  lr * vd / (np.sqrt(sd) + self.epsilon)
        
        return base_pred
    
    def loss(self, y, a):
        return (y - a)**2
    
    def grad(self, y, a):
        # for 0.5 * (y - a)**2
        return a - y
    
    def hess(self, y, a):
        # for 0.5 * (y - a)**2
        return np.full((y.shape), 1)
    

    
class XGBRegressorRMS:
    def __init__(self, eta = 0.3, n_estimators = 100, max_depth = 6, gamma = 0, min_child_weight = 1, lambda_ = 1, colsample = 1, subsample = 1):
        self.eta = eta
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.lambda_ = lambda_
        self.colsample = colsample
        self.subsample = subsample
        self.history = {
            "train" : list(),
            "test" : list()
        }
        
        # list of all weak learners
        self.trees = list()
        
        self.base_pred = None
        
        # adam params
        self.b1 = 0.9
        self.b2 =0.999
        self.epsilon = 1e-7
        
        
    
    def fit(self, x, y, eval_set = None):
        # checking Datatypes
        if isinstance(x, pd.DataFrame) or isinstance(x, pd.Series):
            x = x.values
        if not isinstance(x, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
            
        if isinstance(y, pd.DataFrame) or isinstance(y, pd.Series):
            y = y.values
        if not isinstance(y, np.ndarray):
            raise TypeError("Input should be pandas Dataframe/Series or numpy array.")
        
        
        
        base_pred = np.full(y.shape, np.mean(y)).astype("float64")
        self.base_pred = np.mean(y)
        sd = np.full(y.shape, 0).astype("float64")
        for n in range(self.n_estimators):
            grad = self.grad(y, base_pred)
            hess = self.hess(y, base_pred)
            estimator = XGBTree(
                x,
                y,
                grad,
                hess,
                depth = self.max_depth,
                gamma = self.gamma,
                min_child_weight = self.min_child_weight,
                lambda_ = self.lambda_,
                colsample = self.colsample,
                subsample = self.subsample
            )
            dw = estimator.predict(x)
            #####
            sd = self.b2 * sd + (1 - self.b2) * (dw**2)
            #sd = sd / np.sqrt(1 - self.b2**(n+1))
            #####
            base_pred = base_pred +  self.eta * dw / (np.sqrt(sd) + self.epsilon)
            self.trees.append(estimator)
            
            if eval_set:
                X = eval_set[0]
                Y = eval_set[1]
                cost = np.sqrt(np.mean(self.loss(Y, self.predict(X))))
                self.history["test"].append(cost)
                print(f"[{n}] validation_set-rmse : {cost}", end="\t")
            
            cost = np.sqrt(np.mean(self.loss(y, base_pred)))
            self.history["train"].append(cost)
            print(f"[{n}] train_set-rmse : {cost}")
            
    def predict(self, x):
        sd = np.full((x.shape[0],), 0).astype("float64")
        base_pred = np.full((x.shape[0],), self.base_pred).astype("float64")
        n = 1
        for tree in self.trees:
            dw = tree.predict(x)
            #####
            sd = self.b2 * sd + (1 - self.b2) * (dw**2)
            #sd = sd / np.sqrt(1 - self.b2**(n+1))
            n += 1
            #####
            base_pred +=  self.eta * dw / (np.sqrt(sd) + self.epsilon)
        
        return base_pred
    
    def loss(self, y, a):
        return (y - a)**2
    
    def grad(self, y, a):
        # for 0.5 * (y - a)**2
        return a - y
    
    def hess(self, y, a):
        # for 0.5 * (y - a)**2
        return np.full((y.shape), 1)
    
        