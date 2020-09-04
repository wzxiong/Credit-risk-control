

# log loss
def loglikelihood(preds, train_data):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    grad = preds - labels
    hess = preds * (1. - preds)
    return grad, hess

def logloss(y_pred, dtrain):
    y_true = train_data.get_label()
    y_true = np.array(y_true)
    score = -sum(y_true*np.log(y_prob)) + -sum((1-y_true)*np.log((1-y_prob)))
    return 'logloss', np.mean(score), False

def binary_error(preds, train_data):
    labels = train_data.get_label()
    return 'error', np.mean(labels != (preds > 0.5)), False

model = lgb.train(params=lgb_params, train_set=dtrain,fobj=loglikelihood,feval = logloss, num_boost_round=200)
evaluation(model)

# add weight to FN 认为好人其实是坏人
def logistic_obj(y_hat, dtrain):
    y = dtrain.get_label()
    p = 1. / (1. + np.exp(-y_hat))
    grad = 4 * p * y + p - 5 * y
    hess = (4 * y + 1) * (p * (1.0 - p))
    return grad, hess

def err_rate(y_hat, dtrain):
    y = dtrain.get_label()
    y_hat = np.clip(y_hat, 10e-7, 1-10e-7)
    loss_fn = y*np.log(y_hat)
    loss_fp = (1.0 - y)*np.log(1.0 - y_hat)
    return 'error', np.sum(-(5*loss_fn+loss_fp))/len(y), False

model = lgb.train(params=lgb_params, train_set=dtrain,fobj=logistic_obj,feval = err_rate, num_boost_round=200)
evaluation(model)

# focal loss
from scipy.misc import derivative

def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a,g = alpha, gamma
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a,g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    return 'focal_loss', np.mean(loss), False
focal_loss = lambda x,y: focal_loss_lgb(x, y, alpha=0.25, gamma=1.)
focal_loss_eval = lambda x,y: focal_loss_lgb_eval_error(x, y, alpha=0.25, gamma=1.)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=focal_loss, feval=focal_loss_eval, num_boost_round=200)
evaluation(model)

# mob rank
def loglikelihood_mob(preds, train_data, mob,alpha):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    mob_grad = np.log(2) * 2**preds / np.log(1+mob)
    mob_hess = np.log(2) * np.log(2) * 2**preds / np.log(1+mob)
    grad = preds - labels + alpha*mob_grad
    hess = preds * (1. - preds) + alpha*mob_hess
    return grad, hess

def logloss_mob(y_pred, dtrain, mob,alpha):
    y_true = train_data.get_label()
    y_true = np.array(y_true)
    score = -sum(y_true*np.log(y_prob)) + -sum((1-y_true)*np.log((1-y_prob)))
    score = score - alpha*2**preds / np.log(1+mob)
    return 'logloss', np.mean(score), False

mob_loss = lambda x,y: loglikelihood_mob(x, y, mob=train['mob'],alpha=0.005)
mob_loss_eval = lambda x,y: logloss_mob(x, y, mob=train['mob'],alpha=0.005)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)

# mob rank correction
def loglikelihood_mob(preds, train_data, mob,alpha,beta):
    labels = train_data.get_label()
    preds = 1. / (1. + np.exp(-preds))
    mob_grad =  np.exp(preds) * preds * (1. - preds) / np.log(1+mob*beta)
    mob_hess = (np.exp(preds) * preds**2 * (1. - preds)**2+np.exp(preds) * preds * (1. - preds)**2-np.exp(preds) * preds**2 * (1. - preds)) / np.log(1+mob*beta)
    grad = preds - labels + alpha*mob_grad
    hess = preds * (1. - preds) + alpha*mob_hess
    score1 = -labels*np.log(preds) + -(1-labels)*np.log((1-preds))
    score2 =  alpha*np.exp(preds) / np.log(1+mob*beta)
    print(np.mean(score1),np.mean(score2))
    print(list(grad)[0],list(hess)[0])
    return grad, hess

def logloss_mob(y_pred, dtrain, mob,alpha,beta):
    y_true = train_data.get_label()
    y_true = np.array(y_true)
    score1 = -y_true*np.log(y_pred) + -(1-y_true)*np.log((1-y_pred))
    score2 =  alpha*np.exp(y_pred) / np.log(1+mob*beta)
    score = score1 + score2
    return 'logloss', np.mean(score), False

def loglikelihood_lgb(y_pred, dtrain, mob, alpha,beta):
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -t*np.log(p) + -(1-t)*np.log((1-p)) + alpha*np.exp(p) / np.log(1+mob*beta)
    p = 1/(1+np.exp(-y_pred))
    print(np.mean(-y_true*np.log(p) + -(1-y_true)*np.log((1-p))),np.mean(alpha*np.exp(p) / np.log(1+mob*beta)))
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

def logloss_lgb(y_pred, dtrain, mob, alpha,beta):
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = - y_true*np.log(p) + -(1-y_true)*np.log((1-p)) + alpha*np.exp(p) / np.log(1+mob*beta)
    return 'log_loss', np.mean(loss), False


mob_loss = lambda x,y: loglikelihood_mob(x, y, mob=train['mob'],alpha=1)
mob_loss_eval = lambda x,y: logloss_mob(x, y, mob=train['mob'],alpha=1)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)

mob_loss = lambda x,y: loglikelihood_mob(x, y, mob=train['mob'],alpha=5)
mob_loss_eval = lambda x,y: logloss_mob(x, y, mob=train['mob'],alpha=5)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)

mob_loss = lambda x,y: loglikelihood_mob(x, y, mob=train['mob'],alpha=0.1)
mob_loss_eval = lambda x,y: logloss_mob(x, y, mob=train['mob'],alpha=0.1)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)

mob_loss = lambda x,y: loglikelihood_lgb(x, y, mob=train['mob'],alpha=0.8,beta=4)
mob_loss_eval = lambda x,y: logloss_lgb(x, y, mob=train['mob'],alpha=0.8,beta=4)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)
#loss 4 (1.489500945632471, [0.31028769101013953, 0.2919977492149251, 0.2899694841258055, 0.307587749650808, 0.28965827163079294])


mob_loss = lambda x,y: loglikelihood_lgb(x, y, mob=train['mob'],alpha=10,beta=4)
mob_loss_eval = lambda x,y: logloss_lgb(x, y, mob=train['mob'],alpha=10,beta=4)
model = lgb.train(params=lgb_params, train_set=dtrain,fobj=mob_loss, feval=mob_loss_eval, num_boost_round=200)
evaluation(model)


