import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np
import pandas as pd
import os,sys,time,datetime
from   sksurv import metrics

def infoLine(message,infoType="info"):
    infoType = infoType.upper()
    if len(infoType) < 5:
        infoType=infoType + " "
    time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    outline = "[" + infoType + " " + str(time) + "] " + message
    print(outline)

    if infoType == "ERROR":
        sys.exit()
    #
    sys.stdout.flush()
#

class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(MaskedLinear, self).__init__(in_features, out_features, bias)

    def set_mask(self, mask):
        self.mask = torch.autograd.Variable(mask, requires_grad=False, volatile=False)
        if torch.cuda.is_available():
            self.weight.data = self.weight.data.cuda()*self.mask.data.cuda()
        else:
            self.weight.data = self.weight.data*self.mask.data

    def forward(self, x):
        weight = self.weight*self.mask
        return F.linear(x, weight, self.bias)
#

class MaskedNet(nn.Module):
    def __init__(self, paramHash):
        super(MaskedNet, self).__init__()
        self.act1 = nn.Tanh() 
        self.act2 = nn.Tanh() 
        self.act3 = nn.Tanh()

        self.pathway_mask = paramHash["Pathway_Mask"]
        self.sc1   = MaskedLinear(paramHash["In_Nodes"], paramHash["Pathway_Nodes"], bias=True)
        self.sc1.set_mask(paramHash["Pathway_Mask"])
        self.sc2   = nn.Linear(paramHash["Pathway_Nodes"], paramHash["net_postPath_1_nodes"], bias=True)
        self.sc3   = nn.Linear(paramHash["net_postPath_1_nodes"], paramHash["net_postPath_2_nodes"], bias=True)

        self.drop1 = nn.Dropout(p=paramHash["net_drop_out"][0], inplace=False)  ## gene layer
        self.drop2 = nn.Dropout(p=paramHash["net_drop_out"][1], inplace=False)  ## pathway layer
        self.drop3 = nn.Dropout(p=paramHash["net_drop_out"][2], inplace=False)  ## postPath_1
        self.drop4 = nn.Dropout(p=paramHash["net_drop_out"][3], inplace=False)  ## postpath_2
        self.clinicunit = paramHash["net_clinic_nodes"]
        ### last layer  +  clinic features --> Cox layer
        if self.clinicunit >=1:
            self.sc4 = nn.Linear(paramHash["net_postPath_2_nodes"]+paramHash["net_clinic_nodes"], 1, bias=False)
        else:
            self.sc4 = nn.Linear(paramHash["net_postPath_2_nodes"], 1,  bias=False)
    def forward(self, x):
        n_row = x.shape[0]
        n_col = x.shape[1]
        x_1 = x
        if self.clinicunit >=1:
            x_1 = x[:,range(0, n_col-self.clinicunit)]
            x_2 = x[:,range(n_col-self.clinicunit, n_col)]
        x_1 = self.drop1(x_1)
        x_1 = self.act1(self.sc1(x_1))
        x_1 = self.drop2(x_1)
        x_1 = self.act2(self.sc2(x_1))
        x_1 = self.drop3(x_1)
        x_1 = self.act3(self.sc3(x_1))
        x_1 = self.drop4(x_1)
        ###combine clinical feature with hidden layer 2 along dimension 1
        if self.clinicunit >=1:
            x_cat = torch.cat((x_1, x_2), dim=1)
        else:
            x_cat = x_1
        lin_pred  = self.sc4(x_cat)
        return lin_pred
#

def R_set(x):
    n_sample = x.size(0)
    matrix_ones = torch.ones(n_sample, n_sample)
    indicator_matrix = torch.tril(matrix_ones)
    return(indicator_matrix)
#

def neg_par_log_likelihood(pred, ytime, yevent):
    ## Cox negative partial log-likelihood.
    n_observed = yevent.sum(0)
    ytime_indicator = R_set(ytime)
    if torch.cuda.is_available():
        ytime_indicator = ytime_indicator.cuda()
    risk_set_sum = ytime_indicator.mm(torch.exp(pred))
    diff = pred - torch.log(risk_set_sum)
    sum_diff_in_observed = torch.transpose(diff, 0, 1).mm(yevent)
    cost = (- (sum_diff_in_observed / n_observed)).reshape((-1,))
    return(cost)
#

def c_index(pred, ytime, yevent):
    ##   Harrell's c-index (between 0 and 1).
    n_sample = yevent.shape[0]
    if torch.cuda.is_available():
        yevent   = np.array(yevent.cuda().detach().cpu().reshape([n_sample, ]), dtype=bool)
        ytime    = ytime.reshape([n_sample, ]).detach().cpu().numpy()
        pred     = pred.reshape([n_sample, ]).detach().cpu().numpy()
    else:
        yevent   = np.array(yevent.reshape([n_sample, ]), dtype=bool)
        ytime    = ytime.reshape([n_sample, ]).numpy()
        pred     = pred.reshape([n_sample, ]).numpy()
    CIndex   = metrics.concordance_index_censored(yevent, ytime, pred)
    return(torch.tensor(CIndex[0]))
#

def evaluate(model, data, paramHash):
    model.eval()
    with torch.no_grad():
        if paramHash["net_clinic_nodes"] >= 1:
            digit  = model(torch.cat((data["x"], data["clinic"]), dim=1))
        else:
            digit  = model(data["x"])
        loss   = neg_par_log_likelihood(digit, data["ytime"], data["yevent"])
        cindex = c_index(digit, data["ytime"], data["yevent"])
    #
    if torch.cuda.is_available():
        return [loss.data.detach().cpu().numpy(), cindex.data.detach().cpu().numpy(), digit.detach().cpu().numpy()]
    else:
        return [loss.data.numpy(), cindex.data.numpy(), digit.numpy()]
#


def trainMaskedNet(train_x, train_clinic, train_ytime, train_yevent, \
                        valid_x, valid_clinic, valid_ytime, valid_yevent, paramHash):
    net = MaskedNet(paramHash)
    if torch.cuda.is_available():
        net.cuda()
    modelfile    = paramHash["model_dir"] + "/model_" + paramHash["prefix"] + ".pt"
    modelfile_ES = paramHash["model_dir"] + "/model_ES_" + paramHash["prefix"] + ".pt"
    list_early_stop    = np.zeros(paramHash["net_max_epoch"]).tolist()
    num_ES             = 0
    list_train_loss    = []
    list_valid_loss    = []
    list_train_cindex  = []
    list_valid_cindex  = []
    best_count = 0 # early stop when there is no improve for # epoches
    best_performance = -1.0  # C-Index as criterion
    opt = optim.Adam(net.parameters(),lr=paramHash["net_learn_rate"],weight_decay=paramHash["net_l2_regularization"])
    for epoch in range(1,paramHash["net_max_epoch"]+1):
        net.train()
        opt.zero_grad() ###reset gradients to zeros
        if paramHash["net_clinic_nodes"]>=1:
            train_pred = net(torch.cat((train_x, train_clinic), dim=1)) ###Forward
        else:   ## if no clinic feature
            train_pred = net(train_x)
        train_loss = neg_par_log_likelihood(train_pred, train_ytime, train_yevent) ###Forward
        train_loss.backward() ###calculate gradients
        opt.step() ### update weights and biases

        trainData = {"x":train_x, "clinic":train_clinic, "ytime":train_ytime, "yevent":train_yevent}
        evaluation_train= evaluate(net, trainData, paramHash)
        
        validData = {"x":valid_x, "clinic":valid_clinic, "ytime":valid_ytime, "yevent":valid_yevent}
        evaluation_valid= evaluate(net, validData, paramHash)
        list_train_loss.append(evaluation_train[0])
        list_train_cindex.append(evaluation_train[1])
        list_valid_loss.append(evaluation_valid[0])
        list_valid_cindex.append(evaluation_valid[1])
        torch.save(net.state_dict(), modelfile)

        if paramHash["net_lr_factor"] < 1.0:
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='min', factor=paramHash["net_lr_factor"], patience=paramHash["net_patience"], verbose=True)

        # check improvement
        improvement = False
        if evaluation_valid[1] > best_performance and epoch > paramHash["net_min_epoch"]:
            best_performance = evaluation_valid[1]
            improvement = True
            if num_ES == 0:  ## save before first time meet patience
                torch.save(net.state_dict(), modelfile_ES)
            best_count = 0
        if epoch > paramHash["net_min_epoch"] and improvement==False:
            best_count = best_count + 1
            if best_count >= paramHash["net_patience"]:
                #infoLine("No improve for " + str(best_count) + " continuous epoches, stop training")
                if num_ES == 0:  ## indicate when first time meet patience
                    list_early_stop[epoch-paramHash["net_patience"]-1] = 1
                num_ES = num_ES + 1
                if paramHash["early_stopping"]:
                    break
        #
    return (list_train_loss, list_train_cindex, list_valid_loss, list_valid_cindex, list_early_stop)
#

def sort_data(path, clinicUnit=1, sort=True):
    data = pd.read_csv(path)
    cols = data.columns.values.tolist()
    if(sort):
        data.sort_values("OS_MONTHS", ascending = False, inplace = True)
    idx  = [ i for i, word in enumerate(cols) if word.startswith("ENSG") ] 

    x      = data.iloc[:,idx].values
    ytime  = data.loc[:, ["OS_MONTHS"]].values
    yevent = data.loc[:, ["OS_EVENTS"]].values
    sampleID = data.loc[:, ["Sample"]].values
    clinic = []
    if clinicUnit >= 1:
        clinicVar = cols[(idx[-1]+1):(idx[-1]+1+clinicUnit)]
        clinic = data.loc[:, clinicVar].values
    else:
        clinic = np.zeros(len(ytime))
    return(x, ytime, yevent, clinic, sampleID)


def load_data(path, clinicUnit=1, dtype="float32"):
    # Load the sorted data, and then covert it to a Pytorch tensor.
    x, ytime, yevent, clinic, SampleID = sort_data(path, clinicUnit=clinicUnit)
    X      = torch.from_numpy(x).type(dtype)
    YTIME  = torch.from_numpy(ytime).type(dtype)
    YEVENT = torch.from_numpy(yevent).type(dtype)
    if clinicUnit >=1:
        Clinic = torch.from_numpy(clinic).type(dtype)
    else:
        Clinic = torch.zeros(clinic.shape).type(dtype)
    if torch.cuda.is_available():
        X = X.cuda()
        YTIME = YTIME.cuda()
        YEVENT = YEVENT.cuda()
        Clinic = Clinic.cuda()
    return(X, YTIME, YEVENT, Clinic, SampleID)
#

def load_data_unsort(path, clinicUnit=1, dtype="float32"):
    x, ytime, yevent, clinic, SampleID = sort_data(path, clinicUnit=clinicUnit, sort=False)
    X      = torch.from_numpy(x).type(dtype)
    YTIME  = torch.from_numpy(ytime).type(dtype)
    YEVENT = torch.from_numpy(yevent).type(dtype)
    if clinicUnit >=1:
        Clinic = torch.from_numpy(clinic).type(dtype)
    else:
        Clinic = torch.zeros(clinic.shape).type(dtype)
    if torch.cuda.is_available():
        X = X.cuda()
        YTIME = YTIME.cuda()
        YEVENT = YEVENT.cuda()
        Clinic = Clinic.cuda()
    return(X, YTIME, YEVENT, Clinic, SampleID)
#


def load_pathway(path, dtype):
    pathway_mask = pd.read_csv(path, index_col = 0).to_numpy() 
    PATHWAY_MASK = torch.from_numpy(pathway_mask).type(dtype)
    if torch.cuda.is_available():
        PATHWAY_MASK = PATHWAY_MASK.cuda()
    return(PATHWAY_MASK)
#


