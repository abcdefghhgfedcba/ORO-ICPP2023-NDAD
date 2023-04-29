import copy
import csv
import json
import os
import time
from heapq import nsmallest
from optparse import Option
from pathlib import Path
# from turtle import mode

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.multiprocessing as mp

import utils.fflow as flw
import wandb
from algorithm.fedbase import BasicClient, BasicServer
from main import logger
from utils.aggregate_funct import *
from sklearn.metrics import *
from sklearn.cluster import KMeans


class MPBasicServer(BasicServer):
    def __init__(self, option, model, clients, test_data=None):
        super().__init__(option, model, clients, test_data)
        self.gpus = option['num_gpus']
        self.num_threads = option['num_threads_per_gpu'] * self.gpus
        self.server_gpu_id = option['server_gpu_id']
        self.log_folder = option['log_folder']
        self.confidence_score = {}
        self.computation_time = {}
        self.type_image = {}
    
    def agg_fuction(self, client_models):
        print(self.agg_option)
        server_model = copy.deepcopy(self.model)
        server_param = []
        for name, param in server_model.state_dict().items():
            server_param=param.view(-1) if not len(server_param) else torch.cat((server_param,param.view(-1)))
        
        client_params = []
        for client_model in client_models:
            client_param=[]
            for name, param in client_model.state_dict().items():
                client_param=param.view(-1) if not len(client_param) else torch.cat((client_param,param.view(-1)))
            client_params=client_param[None, :] if len(client_params)==0 else torch.cat((client_params,client_param[None,:]), 0)
        
        for idx, client in enumerate(client_params):
            client_params[idx] = torch.sub(client, server_param)

        if self.agg_option=='median':
            agg_grads=torch.median(client_params,dim=0)[0]

        elif self.agg_option=='mean':
            agg_grads=torch.mean(client_params,dim=0)

        elif self.agg_option=='trmean':
            ntrmean = 2
            agg_grads=tr_mean(client_params, ntrmean)

        elif self.agg_option=='krum' or self.agg_option=='mkrum' or self.agg_option=='mkrum3':
            multi_k = False if self.agg_option == 'krum' else True
            print('multi krum is ', multi_k)
            if self.agg_option=='mkrum3': 
                nkrum = 3
            else: 
                nkrum = 1
            agg_grads, krum_candidate = multi_krum(client_params, nkrum, multi_k=multi_k)
            
        elif self.agg_option=='bulyan' or self.agg_option=='bulyan2':
            if self.agg_option=='bulyan': 
                nbulyan = 1
            elif self.agg_option=='bulyan2': 
                nbulyan = 2
            agg_grads, krum_candidate=bulyan(client_params, nbulyan)

        start_idx=0
        model_grads=[]
        new_global_model = copy.deepcopy(self.model)
        for name, param in new_global_model.state_dict().items():
            param_=agg_grads[start_idx:start_idx+len(param.data.view(-1))].reshape(param.data.shape)
            start_idx=start_idx+len(param.data.view(-1))
            param_=param_.cuda()
            new_global_model.state_dict()[name].copy_(param + param_)
            # model_grads.append(param_)
            
        return new_global_model
    def run(self):
        """
        Start the federated learning symtem where the global model is trained iteratively.
        """
        pool = mp.Pool(self.num_threads)
        logger.time_start('Total Time Cost')
        for round in range(1, self.num_rounds + 1):
            self.current_round = round
            print("--------------Round {}--------------".format(round))
            logger.time_start('Time Cost')

            self.iterate(round, pool)
            
            if round == self.num_rounds:
                path_js = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                
                for client in self.clients:
                    self.type_image[client.name] = client.train_data.type_image_idx   
                # with open(path_js + 'type_image.json', 'w') as json_file:
                #     json.dump(self.type_image, json_file, indent=4)
            # decay learning rate
            self.global_lr_scheduler(round)

            logger.time_end('Time Cost')
            if logger.check_if_log(round, self.eval_interval): logger.log(self)
            print('Max test acc = ',max(logger.output['test_accs']))
            path_save_model = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/model.pt' 
            torch.save(self.model.state_dict(), path_save_model)
        print("=================End==================")
        logger.time_end('Total Time Cost')
        # # save results as .json file
        filepath = os.path.join(self.log_folder, 'log/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'])
        if not Path(filepath).exists():
            os.system(f"mkdir -p {filepath}")
        logger.save(os.path.join(filepath, 'logger.json'))
        # logger.save(os.path.join(filepath, flw.output_filename(self.option, self)))
        
    def iterate(self, round, pool):
        """
        The standard iteration of each federated round that contains three
        necessary procedure in FL: client selection, communication and model aggregation.
        :param
            t: the number of current round
        """
        # sample clients: MD sampling as default but with replacement=False
        self.selected_clients = self.sample()
        models, acc_before_trains, loss_before_trains, confidence_score_dict, calculate_cs_time, train_time = self.communicate(self.selected_clients, pool)
        self.computation_time[round] = {}
        if self.option["agg_algorithm"] == "NDAD":
            self.computation_time[round]["calculate_cs_time"] = calculate_cs_time
            self.computation_time[round]["train_time"] = train_time
        else:
            self.computation_time[round]["train_time"] = train_time
            
            
        self.confidence_score[round] = {}
        # for client in self.clients:
        for idx,client in enumerate(self.selected_clients):
            self.confidence_score[round][int(client)] = confidence_score_dict[idx]
            
        peers_types = [self.clients[id].train_data.client_type for id in self.selected_clients]
        path_js = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
        if not os.path.exists(path_js):
            os.makedirs(path_js)
        # with open(path_js + 'confidence_score.json', 'w') as json_file:
        #     json.dump(self.confidence_score, json_file, indent=4)
            
        # check whether all the clients have dropped out, because the dropped clients will be deleted from self.selected_clients
        if not self.selected_clients: return
        # aggregate: pk = 1/K as default where K=len(selected_clients)
        device0 = torch.device(f"cuda:{self.server_gpu_id}")
        models = [i.to(device0) for i in models]
        # self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
        # if self.current_round != -1:
        if self.agg_option == 'mean':
            if self.option['agg_algorithm'] != "fedavg":  
                ours_server_time_start = time.time()
                list_peak = []
                list_confidence_score = []
                
                df_round = pd.DataFrame()
                for idx,client in enumerate(self.selected_clients):
                        confidence_score_dict_client = self.confidence_score[round][int(client)]
                        df_client = pd.DataFrame.from_dict(confidence_score_dict_client, orient='index', columns=['Cs'])
                        ax = sns.displot(df_client, x='Cs', kind="kde")
                        for ax in ax.axes.flat:
                            # print (ax.lines)
                            for line in ax.lines:
                                x = line.get_xdata() # Get the x data of the distribution
                                y = line.get_ydata() # Get the y data of the distribution
                        maxid = np.argmax(y) 
                        list_peak.append(y[maxid])
                        list_confidence_score.append(df_client.Cs.mean())
                        plt.close()
                        df_round = pd.concat([df_round, df_client])

                mean_cs_global = sum(list_confidence_score)/len(list_confidence_score)
                ax = sns.displot(df_round, x='Cs', kind="kde")
                for ax in ax.axes.flat:
                            # print (ax.lines)
                    for line in ax.lines:
                        x = line.get_xdata() # Get the x data of the distribution
                        y = line.get_ydata() # Get the y data of the distribution
                maxid = np.argmax(y) 
                peak_global = y[maxid]
                plt.close()
                predicted_normal = []
                predicted_attacker = []
                list_peak_normal = []
                list_peak_attacker = []
                list_cs_normal = []
                list_cs_attacker = []
                for idx, client in enumerate(self.selected_clients):
                    if self.option['agg_algorithm'] == "peak_only": 
                        if list_peak[idx] > peak_global:
                            predicted_attacker.append(client)
                            list_peak_attacker.append(list_peak[idx])
                            list_cs_attacker.append(list_confidence_score[idx])
                        else:
                            predicted_normal.append(client)
                            list_peak_normal.append(list_peak[idx])
                            list_cs_normal.append(list_confidence_score[idx])
                    elif self.option['agg_algorithm'] == "cs_only":
                        if list_confidence_score[idx] < mean_cs_global:
                            predicted_attacker.append(client)
                            list_peak_attacker.append(list_peak[idx])
                            list_cs_attacker.append(list_confidence_score[idx])
                        else:
                            predicted_normal.append(client)
                            list_peak_normal.append(list_peak[idx])
                            list_cs_normal.append(list_confidence_score[idx])
                    if self.option['agg_algorithm'] == "NDAD":
                        if list_confidence_score[idx] < mean_cs_global and list_peak[idx] > peak_global:
                            predicted_attacker.append(client)
                            list_peak_attacker.append(list_peak[idx])
                            list_cs_attacker.append(list_confidence_score[idx])
                        else:
                            predicted_normal.append(client)
                            list_peak_normal.append(list_peak[idx])
                            list_cs_normal.append(list_confidence_score[idx])

                if self.option['agg_algorithm'] in ["NDAD",
                                                    "peak_only",
                                                    "cs_only"]:
                    p_ = []
                    sum_sample = sum([self.client_vols[cid] for cid in self.selected_clients])
                    id_attacker = 0
                    for client in self.selected_clients:
                        if client in predicted_normal:
                            p_.append(self.client_vols[client]/sum_sample)
                        else:
                            max_cs = max(list_confidence_score)
                            min_cs = min(list_confidence_score)
                            csi = (list_cs_attacker[id_attacker] - min_cs)/(max_cs-min_cs)
                            p_.append((csi+self.option['ours_beta']*min_cs/max_cs) * self.client_vols[client]/sum_sample)
                            id_attacker +=1
                
                if len(predicted_normal) > 0:
                    self.model = self.aggregate(models, p = p_)
                ours_server_time = time.time() - ours_server_time_start
                self.computation_time[round]['server_aggregation_time'] = ours_server_time
                
                attacker_idx = []
                for idx, client in enumerate(self.selected_clients):
                    # if Y[y] == "attacker":
                    if self.clients[client].train_data.client_type == "attacker":
                        attacker_idx.append(client)
                # attacker_idx = np.nonzero(Y_)[0]        
                print(f"Attacker idx: {attacker_idx}")
                print(f"Predicted attacker: {predicted_attacker}")
                    
                
                true_pred = list(set(attacker_idx) & set(predicted_attacker))
                print("True prediction attackers: {}/{}".format(len(true_pred),len(attacker_idx)))
                wrong_pred_attacker = []
                for client in predicted_attacker:
                    if client not in attacker_idx:
                        wrong_pred_attacker.append(client)
                print("Wrong prediction attackers: {}/{}".format(len(wrong_pred_attacker),len(predicted_attacker)))
                wrong_pred_normal = []
                for client in predicted_normal:
                    if client in attacker_idx:
                        wrong_pred_normal.append(client)
                print("Wrong prediction normals: {}/{}".format(len(wrong_pred_normal),len(predicted_normal)))
                print('\n')
                
                dictionary = {
                "Real attacker": [int(i) for i in attacker_idx],
                "Global peak": peak_global,
                "List peak predicted normal": list_peak_normal,
                "List peak predicted attacker": list_peak_attacker,
                "Global cs": mean_cs_global,
                "List cs predicted normal": list_cs_normal,
                "List cs predicted attacker": list_cs_attacker,
                "Predicted normal": [int(i) for i in predicted_normal],
                "Predicted attacker": [int(i) for i in predicted_attacker],
                "true prediction attacker": [int(i) for i in true_pred],
                "wrong prediction attacker": [int(i) for i in wrong_pred_attacker],
                "wrong prediction normal": [int(i) for i in wrong_pred_normal],
                }
                path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                listObj = []
                if round != 1:
                    with open(path_ + 'log.json') as fp:
                        listObj = json.load(fp)
                
                listObj.append(dictionary)
                
                with open(path_ + 'log.json', 'w') as json_file:
                    json.dump(listObj, json_file, indent=4)
                if self.option['log_time'] == 1:
                    with open(path_ + 'log_time.json', 'w') as f:
                        json.dump(self.computation_time, f, indent=4)
                
            else: #Fedavg
                avg_confidence_score_list = []
                for idx,client in enumerate(self.selected_clients):
                    # if idx in list_idx:
                    confidence_score_dict_client = self.confidence_score[round][int(client)]
                    avg_confidence_score_list.append(sum(confidence_score_dict_client.values())/len(confidence_score_dict_client))
                all_client = {}
                for id,client in enumerate(self.selected_clients):
                    all_client[int(client)] = [float(self.clients[client].train_data.dirty_rate), float(loss_before_trains[id]), float(acc_before_trains[id]), float(avg_confidence_score_list[id])]
                print("client_rate", all_client)
                path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
                
                fedavg_time_start = time.time()
                self.model = self.aggregate(models, p = [1.0 * self.client_vols[cid]/self.data_vol for cid in self.selected_clients])
                fedavg_time = time.time() - fedavg_time_start
                self.computation_time[round]['server_aggregation_time'] = fedavg_time
                
                if self.option['log_time'] == 1:
                    with open(path_ + 'log_time.json', 'w') as f:
                        json.dump(self.computation_time, f, indent=4)
            
            print(f'Done aggregate at round {self.current_round}')
        else: #other defense
            other_defense_time_start = time.time()
            self.model = self.agg_fuction(models)
            other_defense_time = time.time() - other_defense_time_start
            self.computation_time[round]['server_aggregation_time'] = other_defense_time
            path_ = self.log_folder + '/' + self.option['task'] + '/' + self.option['noise_type'] + '/' + 'num_malicious_{}/dirty_rate_{}/attacked_class_{}/'.format( self.option['num_malicious'], self.option['dirty_rate'][0], len(self.option['attacked_class'])) + self.option['agg_algorithm'] + '/'
            if self.option['log_time'] == 1:
                with open(path_ + 'log_time.json', 'w') as f:
                    json.dump(self.computation_time, f, indent=4)
                
            print(f'Done aggregate at round {self.current_round}')
            
    def communicate(self, selected_clients, pool):
        """
        The whole simulating communication procedure with the selected clients.
        This part supports for simulating the client dropping out.
        :param
            selected_clients: the clients to communicate with
        :return
            :the unpacked response from clients that is created ny self.unpack()
        """
        packages_received_from_clients = []        
        packages_received_from_clients = pool.map(self.communicate_with, selected_clients)
        self.selected_clients = [selected_clients[i] for i in range(len(selected_clients)) if packages_received_from_clients[i]]
        
        packages_received_from_clients = [pi for pi in packages_received_from_clients if pi]
        return self.unpack(packages_received_from_clients)

    def communicate_with(self, client_id):
        """
        Pack the information that is needed for client_id to improve the global model
        :param
            client_id: the id of the client to communicate with
        :return
            client_package: the reply from the client and will be 'None' if losing connection
        """
        
        gpu_id = int(mp.current_process().name[-1]) - 1
        gpu_id = gpu_id % self.gpus

        torch.manual_seed(0)
        torch.cuda.set_device(gpu_id)
        device = torch.device('cuda') # This is only 'cuda' so its can find the propriate cuda id to train
        # package the necessary information
        svr_pkg = self.pack(client_id)
        # listen for the client's response and return None if the client drops out
        if self.clients[client_id].is_drop(): return None
        return self.clients[client_id].reply(svr_pkg, device)


    def test(self, model=None, device=None):
        """
        Evaluate the model on the test dataset owned by the server.
        :param
            model: the model need to be evaluated
        :return:
            the metric and loss of the model on the test data
        """
        if model==None: 
            model=self.model
        if self.test_data:
            model.eval()
            loss = 0
            eval_metric = 0
            inference_metric = 0
            data_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
            for batch_id, batch_data in enumerate(data_loader):
                bmean_eval_metric, bmean_loss, inference_time = self.calculator.test(model, batch_data, device)
                loss += bmean_loss * len(batch_data[1])
                eval_metric += bmean_eval_metric * len(batch_data[1])
                inference_metric += inference_time
            eval_metric /= len(self.test_data)
            loss /= len(self.test_data)
            inference_metric /= len(self.test_data)
            return eval_metric, loss, inference_metric
        else: 
            return -1, -1, -1
    def test_label_predictions(self, model, device):
        model.eval()
        actuals = []
        predictions = []
        test_loader = self.calculator.get_data_loader(self.test_data, batch_size=64)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                prediction = output.argmax(dim=1, keepdim=True)
                
                actuals.extend(target.view_as(prediction))
                predictions.extend(prediction)
        return [i.item() for i in actuals], [i.item() for i in predictions]
    
    def test_on_clients(self, round, dataflag='valid', device='cuda'):
        """
        Validate accuracies and losses on clients' local datasets
        :param
            round: the current communication round
            dataflag: choose train data or valid data to evaluate
        :return
            evals: the evaluation metrics of the global model on each client's dataset
            loss: the loss of the global model on each client's dataset
        """
        evals, losses = [], []
        for c in self.clients:
            eval_value, loss, confidence_score_dict = c.test(self.model, dataflag, device=device)
            evals.append(eval_value)
            losses.append(loss)
        return evals, losses

class MPBasicClient(BasicClient):
    def __init__(self, option, name='', train_data=None, valid_data=None):
        super().__init__(option, name, train_data, valid_data)

    def train(self, model, device, current_round=0):
        """
        Standard local training procedure. Train the transmitted model with local training dataset.
        :param
            model: the global model
            device: the device to be trained on
        :return
        """
        model = model.to(device)
        if self.uncertainty == 0:
            model.train()
            
            print(len(self.train_data.idxs))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
            
            for iter in range(self.epochs):
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    loss = self.calculator.get_loss_not_uncertainty(model, batch_data, device)
                    loss.backward()
                    optimizer.step()
                    
        else:
            model.train()
            print(len(self.train_data.idxs))
            data_loader = self.calculator.get_data_loader(self.train_data, batch_size=self.batch_size)
            
            optimizer = self.calculator.get_optimizer(self.optimizer_name, model, lr = self.learning_rate, weight_decay=self.weight_decay, momentum=self.momentum)
                
            for iter in range(self.epochs):
                uncertainty = 0.0
                total_loss = 0.0
                for batch_id, batch_data in enumerate(data_loader):
                    model.zero_grad()
                    loss, unc = self.calculator.get_loss(model, batch_data, iter, device)
                    loss.backward()
                    optimizer.step() 
                    uncertainty += unc.cpu().detach().numpy() 
                    total_loss += loss
                uncertainty = uncertainty / len(data_loader.dataset)
                total_loss /= len(self.train_data)
                


    def test(self, model, dataflag='valid', device='cpu'):
        """
        Evaluate the model with local data (e.g. training data or validating data).
        :param
            model:
            dataflag: choose the dataset to be evaluated on
        :return:
            eval_metric: task specified evaluation metric
            loss: task specified loss
        """
        # dataset = self.train_data if dataflag=='train' else self.valid_data
        confidence_score_dict = {}
        dataset = self.train_data
        model = model.to(device)
        model.eval()
        loss = 0
        eval_metric = 0
        data_loader = self.calculator.get_data_loader(dataset, batch_size=64)
        for batch_id, batch_data in enumerate(data_loader):
            bmean_eval_metric, bmean_loss, _,confidence_score_list, idx_list  = self.calculator.test_client(model, batch_data, device)
            loss += bmean_loss * len(batch_data[1])
            eval_metric += bmean_eval_metric * len(batch_data[1])
            for i,idx in enumerate(idx_list):
                confidence_score_dict[idx] = confidence_score_list[i]
        eval_metric =1.0 * eval_metric / len(dataset)
        loss = 1.0 * loss / len(dataset)
        return eval_metric, loss, confidence_score_dict


    def reply(self, svr_pkg, device):
        """
        Reply to server with the transmitted package.
        The whole local procedure should be planned here.
        The standard form consists of three procedure:
        unpacking the server_package to obtain the global model,
        training the global model, and finally packing the improved
        model into client_package.
        :param
            svr_pkg: the package received from the server
        :return:
            client_pkg: the package to be send to the server
        """
        model, round = self.unpack(svr_pkg)
        calculate_cs_time_start = time.time()
        acc_before_train, loss_before_train,confidence_score_dict = self.test(model, device)
        calculate_cs_time = time.time() - calculate_cs_time_start
        
        train_time_start = time.time()
        self.train(model, device, round)
        train_time = time.time() - train_time_start
        cpkg = self.pack(model, acc_before_train, loss_before_train, confidence_score_dict, calculate_cs_time, train_time)
        
        return cpkg

    def calculate_unc_all_samples(self, global_model, current_round):
        global_model.eval()
        uncertainty_dict = {}
        output_dict = {}
        for i in range(len(self.train_data)):
            data, label = self.train_data[i]
            uncertainty, output = self.calculator.get_uncertainty(global_model, data)
            uncertainty_dict[self.train_data.idxs[i]] = uncertainty.item()
            output_dict[self.train_data.idxs[i]] = output.item()

        PATH = 'results/uncertainty_all_samples/{}/{}/{}'.format(self.option['model'],self.option['file_save_model'], current_round)
        if not os.path.exists(PATH):
            os.makedirs(PATH)
        with open(PATH + f'/{self.name}.json', 'w') as f:
            json.dump(uncertainty_dict, f)
        with open(PATH + f'/{self.name}_output.json', 'w') as f:
            json.dump(output_dict, f)

    def train_loss(self, model, device):
        """
        Get the task specified loss of the model on local training data
        :param model:
        :return:
        """
        return self.test(model,'train', device)[1]
