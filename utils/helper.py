import numpy as np
import jsonlines as jsonl
from scipy.io import loadmat
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from scipy.stats import pearsonr
import itertools

def reshape_activations(activations,layer):
    reshaped_act = {}
    for layerind in layer:
        if int(layerind) < 30:
            a,b,c,d = activations[layerind].shape
            reshaped_act[layerind] = activations[layerind].reshape(a*b,c*d)

        if int(layerind) > 30:
            a,b,c = activations[layerind].shape
            reshaped_act[layerind] = activations[layerind].reshape(a*b,c)
    return reshaped_act
    
def convertIdStim(stim):

    tripletId = [np.floor(stm / 5).astype(int) for stm in stim]
    
    return tripletId



def or_effect(indTriplets, all_act):
    nStimIds = 40 * 5
    stim_1 = np.zeros(len(indTriplets), dtype=int); stim_2= np.zeros(len(indTriplets), dtype=int) ; stim_3 = np.zeros(len(indTriplets), dtype=int) 
    tripletId = np.zeros((len(indTriplets),3))
    acc_cnnA = 0
    
    for i in range(len(indTriplets)):
        stim_1[i], stim_2[i], stim_3[i] = np.unravel_index(indTriplets[i] - 1, (nStimIds, nStimIds, nStimIds), order='F')
        stim_1[i] = int(stim_1[i]); stim_2[i] = int(stim_2[i]); stim_3[i] = int(stim_3[i])
        tripletId[i] = convertIdStim([stim_1[i], stim_2[i], stim_3[i]])
        if all_act[stim_1[i],stim_2[i]] > all_act[stim_1[i],stim_3[i]]:
            choice = 0
        elif all_act[stim_1[i],stim_2[i] ] <= all_act[stim_1[i],stim_3[i]]:
            choice = 1
        if tripletId[i][0] == tripletId[i][1] and choice == 0:
            acc_cnnA += 1
        elif tripletId[i][0] == tripletId[i][2] and choice == 1:
            acc_cnnA += 1

    
    ave_acc = acc_cnnA/len(indTriplets)
    return ave_acc

def get_pvalue(difs):
    pvalue_right = np.sum(difs > 0)/ len(difs)
    pvalue_left = np.sum(difs < 0)/ len(difs)
    pvalue = 2 * min(pvalue_right,pvalue_left)
    return pvalue



def get_prob_A(tripletId, p_corr):
    target, bA, bB = tripletId
    if target==bA:
        prob_ba = p_corr
    elif target==bB:
        prob_ba = 1 - p_corr
    
    return prob_ba




def get_prob_cnn_A(stim_1, stim_2, stim_3, all_act):
    sim_A = all_act[stim_1,stim_2] # the original version
    sim_B = all_act[stim_1,stim_3] # the original version
    prob_cnnA = (sim_A / (sim_A + sim_B)) # the original version
    # sim_A = all_act[stim_1,stim_2]
    # sim_B = all_act[stim_1,stim_3]
    # min_similarities = np.abs(np.min(np.min(all_act,axis=0)))
    # max_similarities = np.max(np.max(all_act,axis=0))
    # sim_A = (sim_A + min_similarities) /(max_similarities + min_similarities)
    # sim_B = (sim_B + min_similarities) /(max_similarities + min_similarities)
    # prob_cnnA = sim_A / (sim_A + sim_B)
    
    return prob_cnnA




def get_prob_behavior_cnn_a(indTriplets, prob_correct, all_act):
    nStimIds = 40 * 5
    stim_1 = np.zeros(len(indTriplets)); stim_2= np.zeros(len(indTriplets)) ; stim_3 = np.zeros(len(indTriplets)) 
    tripletId = np.zeros((len(indTriplets),3))
    prob_A_behavior = np.zeros(len(indTriplets))
    prob_cnnA = np.zeros(len(indTriplets))
    
    for i in range(len(indTriplets)):
        stim_1[i], stim_2[i], stim_3[i] = np.unravel_index(indTriplets[i] - 1, (nStimIds, nStimIds, nStimIds), order='F')
        tripletId[i] = convertIdStim([stim_1[i], stim_2[i], stim_3[i]])
        prob_A_behavior[i] = get_prob_A(tripletId[i], prob_correct[i])
        prob_cnnA[i] = get_prob_cnn_A(stim_1[i].astype(int), stim_2[i].astype(int), stim_3[i].astype(int),all_act)

    return prob_A_behavior , prob_cnnA



def get_pred_task_dual_net(network_name, lesion_name, version_name, task_sort_name, task_nonsort_name, layer):
    
    filename = './lesioning/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION' + version_name + '/EVALUATION_TASK_' + task_sort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'   

    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        #print(key)
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    
    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    
    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true = np.zeros(shape=(3,y_true.shape[0]))
    task_0_pred = np.zeros(shape=(3,y_true.shape[0]))

    task_0_true[0,:] = y_true
    task_0_pred[0,:] = y_pred

    perc = '0.20'

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true[1,:] = y_true
    task_0_pred[1,:] = y_pred

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_0_true[2,:] = y_true
    task_0_pred[2,:] = y_pred


    ### task 1 ###

    filename = './lesioning/' + network_name + '/' + lesion_name + '/drop_percents_records/VERSION' + version_name + '/EVALUATION_TASK_' + task_nonsort_name  + '/predictions/PARAM_GROUP/predictions.jsonl'  # Elaheh
    
    perc = '0.00'

    reader = jsonl.open(filename)
    pred_data = {}
    for obj in reader:
        key = list(obj.keys())[0]
        pred_data[key]=obj[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true = np.zeros(shape=(3,y_true.shape[0]))
    task_1_pred = np.zeros(shape=(3,y_true.shape[0]))

    task_1_true[0,:] = y_true
    task_1_pred[0,:] = y_pred

    perc = '0.20'

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_nonsort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true[1,:] = y_true
    task_1_pred[1,:] = y_pred

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_true'
    if key in pred_data:
        y_true = pred_data[key]

    key = 'INDEX_' + layer + '/SORTEDBY_' + task_sort_name + '/PERCENT_' + perc + '/y_pred'
    if key in pred_data:
        y_pred = pred_data[key]

    y_true = np.asarray(y_true)
    y_pred =  np.asarray(y_pred)

    task_1_true[2,:] = y_true
    task_1_pred[2,:] = y_pred
    
    return task_0_true, task_0_pred, task_1_true, task_1_pred


def get_bootstrap_prop_acc(task_true, task_pred):

    rand_classes = np.unique(task_true[0,:])
    n_classes = rand_classes.shape[0]

    n_boot = 10000

    prop_acc_boot_1 = np.zeros(shape=(n_boot,))
    prop_acc_boot_2 = np.zeros(shape=(n_boot,))

    for iBoot in range(n_boot):

        if np.mod(iBoot,100) == 0:
            print('iteration: ' + str(iBoot))
        y_true_boot = []
        y_pred_boot_1 = []
        y_pred_boot_2 = []
        y_pred_base_boot = []

        # bootstrap classes
        bootstrap_classes = np.random.choice(rand_classes, size=n_classes, replace=True)

        for i,iClass in enumerate (bootstrap_classes):
            class_images = np.where(task_true[0,:]==iClass)[0]

            bootstrap_images = np.random.choice(class_images, size=class_images.shape[0], replace=True)

            if i==1:
                #print(class_images)
                #print(bootstrap_images)
                y_true_boot = np.ones(shape=(class_images.shape[0],))*iClass
                y_pred_base_boot = task_pred[0,bootstrap_images]
                y_pred_boot_1 = task_pred[1,bootstrap_images]
                y_pred_boot_2 = task_pred[2,bootstrap_images]
            else:
                y_true_boot = np.hstack((y_true_boot, np.ones(shape=(class_images.shape[0],))*iClass))
                y_pred_base_boot = np.hstack((y_pred_base_boot, task_pred[0,bootstrap_images]))
                y_pred_boot_1 = np.hstack((y_pred_boot_1, task_pred[1,bootstrap_images]))
                y_pred_boot_2 = np.hstack((y_pred_boot_2, task_pred[2,bootstrap_images]))

        acc_1 = np.where(y_true_boot == y_pred_boot_1)[0].shape[0]/y_true_boot.shape[0]
        acc_2 = np.where(y_true_boot == y_pred_boot_2)[0].shape[0]/y_true_boot.shape[0]
        acc_base = np.where(y_true_boot == y_pred_base_boot)[0].shape[0]/y_true_boot.shape[0]
        # print(acc)

        prop_acc_boot_1[iBoot] = (acc_base - acc_1)/(acc_base)
        prop_acc_boot_2[iBoot] = (acc_base - acc_2)/(acc_base)

    print(np.mean(prop_acc_boot_1))
    print(np.std(prop_acc_boot_1))

    print(np.mean(prop_acc_boot_2))
    print(np.std(prop_acc_boot_2))
    
    return prop_acc_boot_1, prop_acc_boot_2



def elbow_point(explained_variance_ratio):
    differences = np.diff(explained_variance_ratio)
    second_derivative = np.diff(differences)
    elbow_point = np.argmax(second_derivative) + 2
    return elbow_point
    
    
    


class ActivationAnalysis:
    def __init__(self, activation_files, layer='35'):
        """
        Initialize ActivationAnalysis with activation files and a specified layer.
        """
        self.layer = layer
        self.activations = self.load_activations(activation_files)
        self.cosine_similarities = self.compute_cosine_similarities()
        self.combined_cosine_similarities = self.compute_cosine_similarities_all()

    def load_activations(self, files):
        """
        Load activation data from pickle files.
        """
        activations = {}
        for label, path in files.items():
            try:
                with open(path, 'rb') as f:
                    activations[label] = pickle.load(f)
            except FileNotFoundError:
                print(f"File not found: {path}")
        return activations

    def compute_cosine_similarities(self):
        """
        Compute cosine similarity for Asian and White faces individually.
        """
        reshaped_activations = {
            key: reshape_activations({self.layer: act}, [self.layer])
            for key, act in self.activations.items()
        }
        return {key: cosine_similarity(data[self.layer]) for key, data in reshaped_activations.items()}
    def compute_cosine_similarities_all(self):
        """
        Compute cosine similarity for concatenated Asian and White faces together.
        """
        # Reshape activations and concatenate them for each pair
        reshaped_activations = {
            key: reshape_activations({self.layer: act}, [self.layer])
            for key, act in self.activations.items()
        }

        # Concatenate activations and compute cosine similarity
        combined_cosine_similarity = {
            key: cosine_similarity(np.concatenate([reshaped_activations[f'{key}_white'][self.layer],
                                                reshaped_activations[f'{key}_asian'][self.layer]], axis=0))
            for key in ['dual', 'sw', 'sa']
        }
        
        return combined_cosine_similarity




class Ore_bootstrap:
    def __init__(self, cosine_similarities, triplets_file_path, num_boots=1000):
        """
        Initialize Ore_bootstrap with cosine similarities, triplet data, and bootstrap parameters.
        The ORE is calculated immediately when the class is instantiated.
        """
        self.cosine_similarities = cosine_similarities
        self.triplets = self.load_triplets(triplets_file_path)
        self.num_boots = num_boots
        self.bootstrap_results = self.perform_bootstrapping()
        self.ore = self.get_ore() 
        # print(self.cosine_similarities) 

    def load_triplets(self, triplets_file_path):
        """
        Load triplet data from a MATLAB file.
        """
        try:
            triplets = loadmat(triplets_file_path)['indTriplets']
            return [triplets[i].item() for i in range(len(triplets))]
        except FileNotFoundError:
            print(f"Triplet file not found: {triplets_file_path}")
            return []

    def get_ore(self):
        """
        Calculate the ORE based on the triplets and cosine similarities.
        """
        ore_results = {}
        for key, cos_sim in self.cosine_similarities.items():
            ore_results[key] = or_effect(self.triplets, cos_sim)
        return ore_results
    
    def perform_bootstrapping(self):
        """
        Perform bootstrapping on the cosine similarity matrices.
        """
        results = {key: np.zeros(self.num_boots) for key in self.cosine_similarities}
        for iBoot in range(self.num_boots):
            if iBoot % 1000 == 0:
                print(f"Bootstrap: {iBoot}/{self.num_boots}")
            sample_indices = np.random.choice(len(self.triplets), size=len(self.triplets), replace=True)
            bootstrap_sample = [self.triplets[idx] for idx in sample_indices]
            for key, sim_matrix in self.cosine_similarities.items():
                results[key][iBoot] = or_effect(bootstrap_sample, sim_matrix)
        return results
    
    


def compute_correlation(cosine_similarity_1, cosine_similarity_2):
    """
    Compute correlation for the given pair of cosine similarities.
    """
    return np.corrcoef(cosine_similarity_1, cosine_similarity_2)[0, 1]

def compute_correlations(cosine_similarities_dict):
    """
    Compute correlations for all combinations of two categories.
    """
    correlations = {}

    categories = ['dual', 'sw', 'sa']
    category_pairs = itertools.combinations(categories, 2)
    for category1, category2 in category_pairs:
         # Compute correlation for White faces to White faces
        cosine_similarity_1 = cosine_similarities_dict[f'{category1}'][:200, :200][np.tril_indices(cosine_similarities_dict[f'{category1}'][:200, :200].shape[0], k=-1)]
        cosine_similarity_2 = cosine_similarities_dict[f'{category2}'][:200, :200][np.tril_indices(cosine_similarities_dict[f'{category2}'][:200, :200].shape[0], k=-1)]
        correlations[f'{category1}_{category2}_w_w'] = compute_correlation(cosine_similarity_1, cosine_similarity_2)

        # Compute correlation for Asian faces to Asian faces
        cosine_similarity_1 = cosine_similarities_dict[f'{category1}'][200:, 200:][np.tril_indices(cosine_similarities_dict[f'{category1}'][200:, 200:].shape[0], k=-1)]
        cosine_similarity_2 = cosine_similarities_dict[f'{category2}'][200:, 200:][np.tril_indices(cosine_similarities_dict[f'{category2}'][200:, 200:].shape[0], k=-1)]
        correlations[f'{category1}_{category2}_a_a'] = compute_correlation(cosine_similarity_1, cosine_similarity_2)

        # Compute correlation for Asian faces to White faces
        cosine_similarity_1 = cosine_similarities_dict[f'{category1}'][:200, 200:][np.tril_indices(cosine_similarities_dict[f'{category1}'][:200, 200:].shape[0], k=-1)]
        cosine_similarity_2 = cosine_similarities_dict[f'{category2}'][:200, 200:][np.tril_indices(cosine_similarities_dict[f'{category2}'][:200, 200:].shape[0], k=-1)]
        correlations[f'{category1}_{category2}_w_a'] = compute_correlation(cosine_similarity_1, cosine_similarity_2)
    
    return correlations

 
 
 
 
 
class BehavioralCNNDataCorr:
    def __init__(self, cosine_similarity_configs, participant_data_configs):
        """
        Initialize the beh_cnn_analysis with cosine similarity data, participant data, and a helper module.

        :param cosine_similarity_configs: Dict containing CNN similarity matrices.
        :param participant_data_configs: Dict containing participant behavioral data.
        """
        self.cosine_similarity_configs = cosine_similarity_configs
        self.participant_data_configs = participant_data_configs
        self.beh_probabilities = self.calculate_cnn_probabilities()[0]
        self.cnn_probabilities = self.calculate_cnn_probabilities()[1]
        self.correlations, self.pvals = self.calculate_correlations()
        self.error_bars, self.corr_boots = self.beh_cnn_boot()
        
    def calculate_cnn_probabilities(self):
        """
        Compute probabilities for all CNN configurations and participant data.
        
        """
        ### Getting probablity of chosing an option in humans
        prob_A_behavior = {}
        prob_A_behavior['wf_hw'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_white']['correct_responses'], self.cosine_similarity_configs['sw'][:200,:200])

        prob_A_behavior['af_hw'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_white']['correct_responses'],  self.cosine_similarity_configs['sw'][200:,200:])

        prob_A_behavior['wf_ha'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_asian']['triplets'], self.participant_data_configs['white_human_asian']['correct_responses'],  self.cosine_similarity_configs['sa'][:200,:200])

        prob_A_behavior['af_ha'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_asian']['triplets'], self.participant_data_configs['asian_human_asian']['correct_responses'],  self.cosine_similarity_configs['sa'][200:,200:])


        ### Getting probablity of chosing a for CNNs
        prob_cnnA = {}
        
        for keys in self.cosine_similarity_configs.keys():
            _ , prob_cnnA[f"wf_ha_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_asian']['triplets'], self.participant_data_configs['white_human_asian']['correct_responses'], self.cosine_similarity_configs[keys][:200,:200])

            _ , prob_cnnA[f"af_ha_{keys}"]= get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_asian']['triplets'], self.participant_data_configs['asian_human_asian']['correct_responses'], self.cosine_similarity_configs[keys][200:,200:])

            _ , prob_cnnA[f"wf_hw_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_white']['correct_responses'], self.cosine_similarity_configs[keys][:200,:200])

            _ , prob_cnnA[f"af_hw_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_white']['correct_responses'], self.cosine_similarity_configs[keys][200:,200:])

        return prob_A_behavior, prob_cnnA
    def calculate_correlations(self, n_boot=1000):
        """
        Compute correlations for all CNN and behavioral probability combinations.
        """
        correlation_coefficient = {}
        p_vals = {}
        rng = np.random.default_rng(seed=0)

        for index in self.cosine_similarity_configs.keys():

            for key in ['af_ha', 'af_hw', 'wf_hw', 'wf_ha']:

                beh = self.beh_probabilities[key]
                cnn = self.cnn_probabilities[f"{key}_{index}"]
                corr, _ = pearsonr(beh, cnn)
                correlation_coefficient[f"{key}_{index}"] = corr
                n = len(beh)
                boot_corrs = []
                for _ in range(n_boot):
                    idx = rng.choice(n, n, replace=True)
                    boot_corrs.append(pearsonr(beh[idx], cnn[idx])[0])

                boot_corrs = np.array(boot_corrs)

                # bootstrap p-value (one-tailed): P(r_boot ≤ 0)
                p_vals[f"{key}_{index}"] = np.mean(boot_corrs <= 0)
        # for index in self.cosine_similarity_configs.keys():
        #     correlation_coefficient[f"af_ha_{index}"], p_vals[f"af_ha_{index}"] = pearsonr(self.beh_probabilities['af_ha'] , self.cnn_probabilities[f"af_ha_{index}"])
        #     correlation_coefficient[f"af_hw_{index}"], p_vals[f"af_hw_{index}"] = pearsonr(self.beh_probabilities['af_hw'] , self.cnn_probabilities[f"af_hw_{index}"])
        #     correlation_coefficient[f"wf_hw_{index}"], p_vals[f"wf_hw_{index}"] = pearsonr(self.beh_probabilities['wf_hw'] , self.cnn_probabilities[f"wf_hw_{index}"])
        #     correlation_coefficient[f"wf_ha_{index}"], p_vals[f"wf_ha_{index}"] = pearsonr(self.beh_probabilities['wf_ha'] , self.cnn_probabilities[f"wf_ha_{index}"])
            
        return correlation_coefficient, p_vals
    # def get_results(self):
    #     """
    #     Retrieve the computed correlations.

    #     :return: Dictionary of correlations.
    #     """
    #     return self.correlations
    
    def beh_cnn_boot(self):
        """
        bootstraps the triplets to get significant measures

        :return: Dictionary of errorbars and correlations 
        """
        ### Prepare data for bootstrapping the triplets

        ## Find common triplets between White and Asian participants
        common_triplets_white_faces = np.intersect1d(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_asian']['triplets'])
        common_triplets_asian_faces = np.intersect1d(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_asian']['triplets'])

        ## Find probability of choosing option A between White and Asian participants for White faces
        common_indices = {}
        common_indices["wf_hw"] = [index for index, value in enumerate(self.participant_data_configs['white_human_white']['triplets']) if value in common_triplets_white_faces]
        common_indices['wf_ha'] = [index for index, value in enumerate(self.participant_data_configs['white_human_asian']['triplets']) if value in common_triplets_white_faces]
        
        common_indices['af_hw'] = [index for index, value in enumerate(self.participant_data_configs['asian_human_white']['triplets']) if value in common_triplets_asian_faces]
        common_indices['af_ha'] = [index for index, value in enumerate(self.participant_data_configs['asian_human_asian']['triplets']) if value in common_triplets_asian_faces]

        prob_A_behavior_common = {}
        for ind in self.beh_probabilities.keys():
            prob_A_behavior_common[ind] = self.beh_probabilities[ind][common_indices[f'{ind}']]

        prob_cnnA_common = {}
        for ind in self.beh_probabilities.keys():
            for keys in self.cosine_similarity_configs.keys():
                prob_cnnA_common[f"{ind}_{keys}"] = self.cnn_probabilities[f"{ind}_{keys}"][common_indices[f'{ind}']]
            # prob_cnnA_common[f"wf_ha_{keys}"] = self.cnn_probabilities[f"wf_ha_{keys}"][common_indices["wf_ha"]]
            # ## Find probability of choosing option A between White and Asian participants for Asian faces
            # prob_cnnA_common[f"af_hw_{keys}"] = self.cnn_probabilities[f"af_hw_{keys}"][common_indices_af_hw]
            # prob_cnnA_common[f"af_ha_{keys}"] = self.cnn_probabilities[f"af_ha_{keys}"][common_indices_af_ha]
        
        ## concatenate all the data for easy bootstrapping
        wf_triplets_beh_hw_ha = np.column_stack((
            common_triplets_white_faces, prob_A_behavior_common['wf_hw'], prob_A_behavior_common['wf_ha'], 
            prob_cnnA_common["wf_hw_sw"], prob_cnnA_common["wf_hw_sa"], prob_cnnA_common["wf_hw_dual"],
            prob_cnnA_common["wf_ha_sw"], prob_cnnA_common["wf_ha_sa"], prob_cnnA_common["wf_ha_dual"]
                                                ))
        ## white faces triplets common between Asian and white participants, 
        # probability of choosing A for White participants,  
        # probability of choosing A for Asian participants, sw,sa,dual CNNs
        af_triplets_beh_hw_ha = np.column_stack((
            common_triplets_asian_faces, prob_A_behavior_common['af_hw'], prob_A_behavior_common['af_ha'],
            prob_cnnA_common["af_hw_sw"], prob_cnnA_common["af_hw_sa"], prob_cnnA_common["af_hw_dual"],
            prob_cnnA_common["af_ha_sw"], prob_cnnA_common["af_ha_sa"], prob_cnnA_common["af_ha_dual"]
                     ))

        num_boots = 10000
        # nStimIds = 40 * 5
        corr_boot = {}
        ## White Faces
        for ind in self.beh_probabilities.keys():
            for keys in self.cosine_similarity_configs.keys():
                # Asian Participants
                corr_boot[f"{ind}_{keys}"] = np.zeros(num_boots)
                # corr_boot_wf_ha_sw = np.zeros(num_boots)
                # corr_boot_wf_ha_dual = np.zeros(num_boots)

                # # White Participants
                # corr_boot_wf_hw_sa = np.zeros(num_boots)
                # corr_boot_wf_hw_sw = np.zeros(num_boots)
                # corr_boot_wf_hw_dual = np.zeros(num_boots)

                # ## Asian Faces

                # # Asian Participants
                # corr_boot_af_ha_sa = np.zeros(num_boots)
                # corr_boot_af_ha_sw = np.zeros(num_boots)
                # corr_boot_af_ha_dual = np.zeros(num_boots)

                # # White Participants
                # corr_boot_af_hw_sa = np.zeros(num_boots)
                # corr_boot_af_hw_sw = np.zeros(num_boots)
                # corr_boot_af_hw_dual = np.zeros(num_boots)



        # for White faces
        for iBoot in range (1,num_boots):
            #print progress
            if iBoot % 1000 == 0:
                print(f'Bootstrap: {iBoot} out of {num_boots}')
            # get triplets
            triplets_boot = np.random.choice(wf_triplets_beh_hw_ha.shape[0], size=wf_triplets_beh_hw_ha.shape[0], replace=True)
            bootstrap_sample = wf_triplets_beh_hw_ha[triplets_boot]

            corr_boot["wf_hw_sa"][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,4])
            corr_boot["wf_hw_sw"][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,3])
            corr_boot['wf_hw_dual'][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,5])
            
            corr_boot['wf_ha_sa'][iBoot], _ = pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,7])
            corr_boot['wf_ha_sw'][iBoot], _ = pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,6])
            corr_boot['wf_ha_dual'][iBoot], _= pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,8])
                
                
                
        # error_bar_white_faces = {key: np.std(value) for key, value in corr_boot.items()} 

        # for Asian faces
        for iBoot in range (1,num_boots):
            #print progress
            if iBoot % 1000 == 0:
                print(f'Bootstrap: {iBoot} out of {num_boots}')
            # get triplets
            triplets_boot = np.random.choice(af_triplets_beh_hw_ha.shape[0], size=af_triplets_beh_hw_ha.shape[0], replace=True)
            bootstrap_sample = af_triplets_beh_hw_ha[triplets_boot]
            
            corr_boot['af_hw_sa'][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,4])
            corr_boot['af_hw_sw'][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,3])
            corr_boot['af_hw_dual'][iBoot], _ = pearsonr(bootstrap_sample[:,1] , bootstrap_sample[:,5])
            
            corr_boot['af_ha_sa'][iBoot], _ = pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,7])
            corr_boot['af_ha_sw'][iBoot], _ = pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,6])
            corr_boot['af_ha_dual'][iBoot], _= pearsonr(bootstrap_sample[:,2] , bootstrap_sample[:,8])
            
            
        error_bars = {key: np.std(value) for key, value in corr_boot.items()}  
        
        return error_bars, corr_boot





 
class BehavioralCNNDataCorralltrial:
    def __init__(self, cosine_similarity_configs, participant_data_configs):
        """
        Initialize the beh_cnn_analysis with cosine similarity data, participant data, and a helper module.

        :param cosine_similarity_configs: Dict containing CNN similarity matrices.
        :param participant_data_configs: Dict containing participant behavioral data.
        """
        self.cosine_similarity_configs = cosine_similarity_configs
        self.participant_data_configs = participant_data_configs
        self.beh_probabilities = self.calculate_cnn_probabilities()[0]
        self.cnn_probabilities = self.calculate_cnn_probabilities()[1]
        self.correlations = self.calculate_correlations()
        self.error_bars, self.corr_boots = self.beh_cnn_boot_alltrials()
        
    def calculate_cnn_probabilities(self):
        """
        Compute probabilities for all CNN configurations and participant data.
        
        """
        ### Getting probablity of chosing an option in humans
        prob_A_behavior = {}
        prob_A_behavior['wf_hw'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_white']['correct_responses'], self.cosine_similarity_configs['sw'][:200,:200])

        prob_A_behavior['af_hw'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_white']['correct_responses'],  self.cosine_similarity_configs['sw'][200:,200:])

        prob_A_behavior['wf_ha'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_asian']['triplets'], self.participant_data_configs['white_human_asian']['correct_responses'],  self.cosine_similarity_configs['sa'][:200,:200])

        prob_A_behavior['af_ha'] , _ = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_asian']['triplets'], self.participant_data_configs['asian_human_asian']['correct_responses'],  self.cosine_similarity_configs['sa'][200:,200:])


        ### Getting probablity of chosing a for CNNs
        prob_cnnA = {}
        
        for keys in self.cosine_similarity_configs.keys():
            _ , prob_cnnA[f"wf_ha_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_asian']['triplets'], self.participant_data_configs['white_human_asian']['correct_responses'], self.cosine_similarity_configs[keys][:200,:200])

            _ , prob_cnnA[f"af_ha_{keys}"]= get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_asian']['triplets'], self.participant_data_configs['asian_human_asian']['correct_responses'], self.cosine_similarity_configs[keys][200:,200:])

            _ , prob_cnnA[f"wf_hw_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_white']['correct_responses'], self.cosine_similarity_configs[keys][:200,:200])

            _ , prob_cnnA[f"af_hw_{keys}"] = get_prob_behavior_cnn_a(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_white']['correct_responses'], self.cosine_similarity_configs[keys][200:,200:])

        return prob_A_behavior, prob_cnnA
    def calculate_correlations(self):
        """
        Compute correlations for all CNN and behavioral probability combinations.
        """
        correlation_coefficient = {}
        p_val = {}
        beh_ha = np.concatenate([self.beh_probabilities['af_ha'],self.beh_probabilities['wf_ha']])
        beh_hw = np.concatenate([self.beh_probabilities['af_hw'], self.beh_probabilities['wf_hw']])
        for index in self.cosine_similarity_configs.keys():
            cnn_hw = np.concatenate([self.cnn_probabilities[f"af_hw_{index}"], self.cnn_probabilities[f"wf_hw_{index}"]])
            cnn_ha = np.concatenate([self.cnn_probabilities[f"af_ha_{index}"], self.cnn_probabilities[f"wf_ha_{index}"]])

            correlation_coefficient[f"hw_{index}"], p_val[f"hw_{index}"] = pearsonr(beh_hw, cnn_hw)
            correlation_coefficient[f"ha_{index}"], p_val[f"ha_{index}"]  = pearsonr(beh_ha, cnn_ha)
            
        return correlation_coefficient 
    # def get_results(self):
    #     """
    #     Retrieve the computed correlations.

    #     :return: Dictionary of correlations.
    #     """
    #     return self.correlations
    
    def beh_cnn_boot_alltrials(self):
        """
        bootstraps the triplets to get significant measures

        :return: Dictionary of errorbars and correlations 
        """
        ### Prepare data for bootstrapping the triplets

        ## Find common triplets between White and Asian participants
        common_triplets_white_faces = np.intersect1d(self.participant_data_configs['white_human_white']['triplets'], self.participant_data_configs['white_human_asian']['triplets'])
        common_triplets_asian_faces = np.intersect1d(self.participant_data_configs['asian_human_white']['triplets'], self.participant_data_configs['asian_human_asian']['triplets'])

        ## Find probability of choosing option A between White and Asian participants for White faces
        common_indices = {}
        common_indices["wf_hw"] = [index for index, value in enumerate(self.participant_data_configs['white_human_white']['triplets']) if value in common_triplets_white_faces]
        common_indices['wf_ha'] = [index for index, value in enumerate(self.participant_data_configs['white_human_asian']['triplets']) if value in common_triplets_white_faces]
        
        common_indices['af_hw'] = [index for index, value in enumerate(self.participant_data_configs['asian_human_white']['triplets']) if value in common_triplets_asian_faces]
        common_indices['af_ha'] = [index for index, value in enumerate(self.participant_data_configs['asian_human_asian']['triplets']) if value in common_triplets_asian_faces]
        
        prob_A_behavior_common = {}
        for ind in self.beh_probabilities.keys():
            prob_A_behavior_common[ind] = self.beh_probabilities[ind][common_indices[f'{ind}']]

        prob_cnnA_common = {}
        for ind in self.beh_probabilities.keys():
            for keys in self.cosine_similarity_configs.keys():
                prob_cnnA_common[f"{ind}_{keys}"] = self.cnn_probabilities[f"{ind}_{keys}"][common_indices[f'{ind}']]
            # prob_cnnA_common[f"wf_ha_{keys}"] = self.cnn_probabilities[f"wf_ha_{keys}"][common_indices["wf_ha"]]
            # ## Find probability of choosing option A between White and Asian participants for Asian faces
            # prob_cnnA_common[f"af_hw_{keys}"] = self.cnn_probabilities[f"af_hw_{keys}"][common_indices_af_hw]
            # prob_cnnA_common[f"af_ha_{keys}"] = self.cnn_probabilities[f"af_ha_{keys}"][common_indices_af_ha]
        
        ## concatenate all the data for easy bootstrapping
        # wf_triplets_beh_hw_ha = np.column_stack((
        #     common_triplets_white_faces, prob_A_behavior_common['wf_hw'], prob_A_behavior_common['wf_ha'], 
        #     prob_cnnA_common["wf_hw_sw"], prob_cnnA_common["wf_hw_sa"], prob_cnnA_common["wf_hw_dual"],
        #     prob_cnnA_common["wf_ha_sw"], prob_cnnA_common["wf_ha_sa"], prob_cnnA_common["wf_ha_dual"]
        #                                         ))
        # ## white faces triplets common between Asian and white participants, 
        # # probability of choosing A for White participants,  
        # # probability of choosing A for Asian participants, sw,sa,dual CNNs
        # af_triplets_beh_hw_ha = np.column_stack((
        #     common_triplets_asian_faces, prob_A_behavior_common['af_hw'], prob_A_behavior_common['af_ha'],
        #     prob_cnnA_common["af_hw_sw"], prob_cnnA_common["af_hw_sa"], prob_cnnA_common["af_hw_dual"],
        #     prob_cnnA_common["af_ha_sw"], prob_cnnA_common["af_ha_sa"], prob_cnnA_common["af_ha_dual"]
        #              ))
        # # Concatenate White and Asian triplets
        # triplets_beh_hw_ha = np.vstack([wf_triplets_beh_hw_ha, af_triplets_beh_hw_ha])
        combined_triplets      = np.concatenate([common_triplets_white_faces, common_triplets_asian_faces])
        combined_beh_hw        = np.concatenate([prob_A_behavior_common['wf_hw'], prob_A_behavior_common['af_hw']])
        combined_beh_ha        = np.concatenate([prob_A_behavior_common['wf_ha'], prob_A_behavior_common['af_ha']])

        combined_cnn_hw_sw     = np.concatenate([prob_cnnA_common["wf_hw_sw"], prob_cnnA_common["af_hw_sw"]])
        combined_cnn_hw_sa     = np.concatenate([prob_cnnA_common["wf_hw_sa"], prob_cnnA_common["af_hw_sa"]])
        combined_cnn_hw_dual   = np.concatenate([prob_cnnA_common["wf_hw_dual"], prob_cnnA_common["af_hw_dual"]])

        combined_cnn_ha_sw     = np.concatenate([prob_cnnA_common["wf_ha_sw"], prob_cnnA_common["af_ha_sw"]])
        combined_cnn_ha_sa     = np.concatenate([prob_cnnA_common["wf_ha_sa"], prob_cnnA_common["af_ha_sa"]])
        combined_cnn_ha_dual   = np.concatenate([prob_cnnA_common["wf_ha_dual"], prob_cnnA_common["af_ha_dual"]])

        # Final combined array
        triplets_beh_hw_ha = np.column_stack((
            combined_triplets,
            combined_beh_hw, combined_beh_ha,
            combined_cnn_hw_sw, combined_cnn_hw_sa, combined_cnn_hw_dual,
            combined_cnn_ha_sw, combined_cnn_ha_sa, combined_cnn_ha_dual
        ))


        num_boots = 10000
        corr_boot = {
            "hw_sw": np.zeros(num_boots),
            "hw_sa": np.zeros(num_boots),
            "hw_dual": np.zeros(num_boots),
            "ha_sw": np.zeros(num_boots),
            "ha_sa": np.zeros(num_boots),
            "ha_dual": np.zeros(num_boots),
        }

        for iBoot in range(num_boots):
            if iBoot % 1000 == 0:
                print(f'Bootstrap: {iBoot} out of {num_boots}')
                
            # Sample triplets with replacement
            triplets_boot = np.random.choice(triplets_beh_hw_ha.shape[0], size=triplets_beh_hw_ha.shape[0], replace=True)
            bootstrap_sample = triplets_beh_hw_ha[triplets_boot]

            # Correlations for HW (column 1 = beh_hw, col 3–5 = CNNs)
            corr_boot["hw_sw"][iBoot], _   = pearsonr(bootstrap_sample[:, 1], bootstrap_sample[:, 3])
            corr_boot["hw_sa"][iBoot], _   = pearsonr(bootstrap_sample[:, 1], bootstrap_sample[:, 4])
            corr_boot["hw_dual"][iBoot], _ = pearsonr(bootstrap_sample[:, 1], bootstrap_sample[:, 5])

            # Correlations for HA (column 2 = beh_ha, col 6–8 = CNNs)
            corr_boot["ha_sw"][iBoot], _   = pearsonr(bootstrap_sample[:, 2], bootstrap_sample[:, 6])
            corr_boot["ha_sa"][iBoot], _   = pearsonr(bootstrap_sample[:, 2], bootstrap_sample[:, 7])
            corr_boot["ha_dual"][iBoot], _ = pearsonr(bootstrap_sample[:, 2], bootstrap_sample[:, 8])

        # Compute error bars as std of bootstrap distributions
        error_bars = {key: np.std(values) for key, values in corr_boot.items()}

        
        return error_bars, corr_boot

