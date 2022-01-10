#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from torch.utils.data import Dataset
from .FormatConverter import transfer_mgf
import os 
import pickle
import re
import csv 
from dataclasses import dataclass
import numpy as np
from .vocab import ProteinVocab
from . import configuration
from tqdm import tqdm
import torch
import warnings

warnings.filterwarnings("ignore")

@dataclass
class DDAFeature:
    mz: float
    z: float
    peptide: list
    scan: str
    mass: float
    
def pad_to_length(data: list, length, pad_token=0.):
  for i in range(length - len(data)):
    data.append(pad_token)
    
def transform_function(mz,intensity):
    if np.random.uniform() < 0.5:
        return mz,intensity
    l = len(mz)
    mz = np.array(mz)
    intensity = np.array(intensity)
    p = np.random.uniform(0.05,0.2)
    m = np.random.uniform(0,1,l)
    m = m > p 
    mz = mz[m].tolist()
    intensity = intensity[m].tolist()
    return mz,intensity
    

def process_peaks(spectrum_mz_list, spectrum_intensity_list, peptide_mass, transform):
  charge = 1.0
  spectrum_intensity_max = np.max(spectrum_intensity_list)
  # charge 1 peptide location
  spectrum_mz_list.append(peptide_mass + charge * configuration.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # N-terminal, b-ion, peptide_mass_C
  # append N-terminal
  mass_N = configuration.mass_N_terminus - configuration.mass_H
  spectrum_mz_list.append(mass_N + charge*configuration.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)
  # append peptide_mass_C
  mass_C = configuration.mass_C_terminus + configuration.mass_H
  peptide_mass_C = peptide_mass - mass_C
  spectrum_mz_list.append(peptide_mass_C + charge*configuration.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  # C-terminal, y-ion, peptide_mass_N
  # append C-terminal
  mass_C = configuration.mass_C_terminus + configuration.mass_H
  spectrum_mz_list.append(mass_C + charge*configuration.mass_H)
  spectrum_intensity_list.append(spectrum_intensity_max)

  #special token
  
  if transform != None:
       spectrum_mz_list, spectrum_intensity_list = transform(spectrum_mz_list,spectrum_intensity_list)   

  #spectrum_mz_list.append(0)
  #spectrum_intensity_list.append(spectrum_intensity_max)    
  point_length = min(configuration.MAX_NUM_PEAK, len(spectrum_mz_list))

  pad_to_length(spectrum_mz_list, configuration.MAX_NUM_PEAK)#, -MZ_MAX)
  pad_to_length(spectrum_intensity_list, configuration.MAX_NUM_PEAK)#, -spectrum_intensity_max)

  spectrum_mz = np.array(spectrum_mz_list, dtype=np.float32)

  neutral_mass = spectrum_mz - charge*configuration.mass_H
  in_bound_mask = np.logical_and(neutral_mass > 0., neutral_mass < configuration.MZ_MAX)
  neutral_mass[~in_bound_mask] = 0.
  # intensity
  spectrum_intensity = np.array(spectrum_intensity_list, dtype=np.float32)
  norm_intensity = spectrum_intensity / spectrum_intensity_max

  order = np.argsort(norm_intensity)[::-1]
  mass_location = neutral_mass[order][:configuration.MAX_NUM_PEAK]
  intensity     = norm_intensity[order][:configuration.MAX_NUM_PEAK]

  return mass_location, intensity, point_length

def get_ion_index(peptide_mass, prefix_mass, direction):
  
  if direction == 0:
    candidate_b_mass = prefix_mass + configuration.mass_ID_np
    candidate_y_mass = peptide_mass - candidate_b_mass
  elif direction == 1:
    candidate_y_mass = prefix_mass + configuration.mass_ID_np
    candidate_b_mass = peptide_mass - candidate_y_mass
  candidate_a_mass = candidate_b_mass - configuration.mass_CO

  # b-ions
  candidate_b_H2O = candidate_b_mass - configuration.mass_H2O
  candidate_b_NH3 = candidate_b_mass - configuration.mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # a-ions
  candidate_a_H2O = candidate_a_mass - configuration.mass_H2O
  candidate_a_NH3 = candidate_a_mass - configuration.mass_NH3
  candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - configuration.mass_H2O
  candidate_y_NH3 = candidate_y_mass - configuration.mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # ion_2
  #~   b_ions = [candidate_b_mass]
  #~   y_ions = [candidate_y_mass]
  #~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  a_ions = [candidate_a_mass,
            candidate_a_H2O,
            candidate_a_NH3,
            candidate_a_plus2_charge1]
  ion_mass_list = b_ions + y_ions + a_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)  # 8 by 26

  # ion locations
  # ion_location = np.ceil(ion_mass * SPECTRUM_RESOLUTION).astype(np.int64) # 8 by 26

  in_bound_mask = np.logical_and(
      ion_mass > 0,
      ion_mass <= configuration.MZ_MAX).astype(np.float32)
  ion_location = ion_mass * in_bound_mask  # 8 by 26, out of bound index would have value 0
  return ion_location.transpose()  # 26 by 8

def current_ion(peptide_mass,prefix_mass):
  candidate_b_mass = prefix_mass
  candidate_y_mass = peptide_mass - candidate_b_mass    
  candidate_a_mass = candidate_b_mass - configuration.mass_CO
  
  # b-ions
  candidate_b_H2O = candidate_b_mass - configuration.mass_H2O
  candidate_b_NH3 = candidate_b_mass - configuration.mass_NH3
  candidate_b_plus2_charge1 = ((candidate_b_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # a-ions
  candidate_a_H2O = candidate_a_mass - configuration.mass_H2O
  candidate_a_NH3 = candidate_a_mass - configuration.mass_NH3
  candidate_a_plus2_charge1 = ((candidate_a_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # y-ions
  candidate_y_H2O = candidate_y_mass - configuration.mass_H2O
  candidate_y_NH3 = candidate_y_mass - configuration.mass_NH3
  candidate_y_plus2_charge1 = ((candidate_y_mass + 2 * configuration.mass_H) / 2
                               - configuration.mass_H)

  # ion_2
  #~   b_ions = [candidate_b_mass]
  #~   y_ions = [candidate_y_mass]
  #~   ion_mass_list = b_ions + y_ions

  # ion_8
  b_ions = [candidate_b_mass,
            candidate_b_H2O,
            candidate_b_NH3,
            candidate_b_plus2_charge1]
  y_ions = [candidate_y_mass,
            candidate_y_H2O,
            candidate_y_NH3,
            candidate_y_plus2_charge1]
  a_ions = [candidate_a_mass,
            candidate_a_H2O,
            candidate_a_NH3,
            candidate_a_plus2_charge1]
  ion_mass_list = b_ions + y_ions + a_ions
  ion_mass = np.array(ion_mass_list, dtype=np.float32)
  in_bound_mask = np.logical_and(
      ion_mass > 0,
      ion_mass <= configuration.MZ_MAX).astype(np.float32)
  ion_location = ion_mass * in_bound_mask 
  return ion_location  

class KnapsackSearcher(object):
    def __init__(self, MZ_MAX, knapsack_file,mass_ID):
        self.knapsack_file = knapsack_file
        self.MZ_MAX = MZ_MAX
        self.knapsack_aa_resolution = configuration.KNAPSACK_AA_RESOLUTION
        self.mass_ID = mass_ID
        self.aa_size  = len(mass_ID)
        if os.path.isfile(knapsack_file):
            print("KnapsackSearcher.__init__(): load knapsack matrix")
            self.knapsack_matrix = np.load(knapsack_file)
        else:
            print("KnapsackSearcher.__init__(): build knapsack matrix from scratch")
            self.knapsack_matrix = self._build_knapsack()

    def _build_knapsack(self):
        max_mass = self.MZ_MAX - configuration.mass_N_terminus - configuration.mass_C_terminus
        max_mass_round = int(round(max_mass * self.knapsack_aa_resolution))
        max_mass_upperbound = max_mass_round + self.knapsack_aa_resolution
        knapsack_matrix = np.zeros(shape=(len(configuration.aa2mass), max_mass_upperbound), dtype=bool)
        for aa_id in tqdm(range(3, len(configuration.aa2mass))):
            mass_aa = int(round(self.mass_ID[aa_id] * self.knapsack_aa_resolution))

            for col in tqdm(range(max_mass_upperbound),leave = False):
                current_mass = col + 1
                if current_mass < mass_aa:
                    knapsack_matrix[aa_id, col] = False
                elif current_mass == mass_aa:
                    knapsack_matrix[aa_id, col] = True
                elif current_mass > mass_aa:
                    sub_mass = current_mass - mass_aa
                    sub_col = sub_mass - 1
                    if np.sum(knapsack_matrix[:, sub_col]) > 0:
                        knapsack_matrix[aa_id, col] = True
                        knapsack_matrix[:, col] = np.logical_or(knapsack_matrix[:, col], knapsack_matrix[:, sub_col])
                    else:
                        knapsack_matrix[aa_id, col] = False
        np.save(self.knapsack_file, knapsack_matrix)
        return knapsack_matrix

    def search_knapsack(self, mass, knapsack_tolerance):
        mass_round = int(round(mass * self.knapsack_aa_resolution))
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        if mass_upperbound < configuration.mass_AA_min_round:
            return []
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = np.flatnonzero(np.any(self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)],
                                                axis=1))
        return candidate_aa_id.tolist()

    def positional_search(self,mass, EOS, knapsack_tolerance = int(0.1 * configuration.KNAPSACK_AA_RESOLUTION)):
        mass_round = int(round(mass * self.knapsack_aa_resolution))
        mass_upperbound = mass_round + knapsack_tolerance
        mass_lowerbound = mass_round - knapsack_tolerance
        
        distance = np.array([0 for i in range(self.aa_size)],dtype = np.float32)
        distance[EOS] = configuration.distance_pdf(mass) #np.exp(-np.abs(mass) * 10)
        mass_lowerbound_col = mass_lowerbound - 1
        mass_upperbound_col = mass_upperbound - 1
        candidate_aa_id = self.knapsack_matrix[:, mass_lowerbound_col:(mass_upperbound_col + 1)]

        m = configuration.distance_matrix.copy()
        m[np.logical_not(candidate_aa_id)] = float('inf')
        m = np.abs(m).min(1)
        distance[3:] = configuration.distance_pdf(m[3:]) #np.exp(- m[3:] * 10)
        return distance



def get_feature(mz_list,intensity_list,feature,self, transform = None):
    peak_location, peak_intensity,point_length = process_peaks(mz_list, intensity_list, feature.mass,transform)
    assert np.max(peak_intensity) < 1.0 + 1e-5
        
    forward_id_input   = self.vocab.encode(feature.peptide)[:-1]
    forward_id_target  = self.vocab.encode(feature.peptide)[1:]
    forward_ion_location_index_list = []
        
    prefix_mass  = 0.
    history_mass = []
    current_ion_location = []
    distances = []
        
    for i, id in enumerate(forward_id_input):
        prefix_mass += self.vocab.aa2mass_num[id]
        history_mass.append([prefix_mass,max(feature.mass - prefix_mass,0)])
            
        distance = self.knap.positional_search(history_mass[-1][1] - self.vocab.aa2mass_num[self.vocab.eos_index], self.vocab.eos_index)
        distances.append(distance[1:])
        #if distance[forward_id_target[i]] < 0.1:
        #    raise SystemExit("invalid PSM")
            
        ion_location = get_ion_index(feature.mass, prefix_mass, 0)
        forward_ion_location_index_list.append(ion_location)
        current_ion_location.append(current_ion(feature.mass, prefix_mass))
        
    return peak_location,\
           peak_intensity,\
           forward_id_target,\
           forward_ion_location_index_list,\
           forward_id_input,\
           history_mass,\
           point_length,\
           current_ion_location,\
           distances     


class DenovoDataset(Dataset):
    
    def __init__(self, spectrum_filename, transform = transform_function, vocab = ProteinVocab(), logfun = print, reuse = True):
        self.spectrum_filename      = spectrum_filename
        self.vocab                  = vocab
        self.input_spectrum_handle  = None
        self.feature_list           = []
        self.spectrum_location_dict = dict()
        self.transform              = transform
        self.logfun                 = logfun
        
        self.knap = KnapsackSearcher(configuration.MZ_MAX,'knapsack.npy', vocab.aa2mass_num)
        
        # read spectrum location file
        spectrum_location_file = spectrum_filename + '.location.pytorch.pkl'
        if os.path.exists(spectrum_location_file) and reuse:
            with open(spectrum_location_file, 'rb') as fr:
                self.spectrum_location_dict = pickle.load(fr)
            self.logfun("Load location file: %s"%(spectrum_location_file))
        else:
            spectrum_location_dict = {}
            line = True
            with open(spectrum_filename, 'r') as f:
                while line:
                    current_location = f.tell()
                    line = f.readline()
                    if "BEGIN IONS" in line:
                        spectrum_location = current_location
                    elif "SCANS=" in line:
                        scan = re.split('[=\r\n]', line)[1]
                        spectrum_location_dict[scan] = spectrum_location
                        
            #map each spectrum to starting line 
            self.spectrum_location_dict = spectrum_location_dict
            with open(spectrum_location_file, 'wb') as fw:
                pickle.dump(self.spectrum_location_dict, fw)     
            self.logfun("Create new location file: %s" % (spectrum_location_file))   
        #check if exists otherwise create one    
        feature_filename = spectrum_filename + ".csv"
        if not os.path.exists(feature_filename) or not reuse:
            transfer_mgf(spectrum_filename, feature_filename)
                
        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        count = {}
        total_num = 0
        with open(feature_filename, "r") as fr:
            reader = csv.reader(fr, delimiter=',')
            header = next(reader)
            mz_index = header.index("m/z")
            z_index = header.index("z")
            seq_index = header.index("seq")
            scan_index = header.index("scans")
            
            for line in reader:
                mass        = (float(line[mz_index]) - configuration.mass_H) * float(line[z_index])
                ok, peptide = self.parse_raw_sequence(line[seq_index])
                
                if not ok:
                    skipped_by_ptm  += 1
                    continue
                if mass > configuration.MZ_MAX:
                    skipped_by_mass   += 1
                    continue
                if len(peptide) >= configuration.MAX_LEN:
                    skipped_by_length += 1
                    continue

                for aa in peptide:
                    count[aa] = count.get(aa, 0) + 1
                    total_num += 1
                new_feature = DDAFeature(mz=float(line[mz_index]),
                                         z=float(line[z_index]),
                                         peptide=peptide,
                                         scan=line[scan_index],
                                         mass=mass)
                
                self.feature_list.append(new_feature)
                
        self.logfun(f"Total number of PTM: {len(self.feature_list)}, "
              f"Invalid amino acid: {skipped_by_ptm}, "
              f"Large mass: {skipped_by_mass}, "
              f"Long sequency: {skipped_by_length} "
            )
        
        for k,v in count.items():
            count[k] = v/total_num
        self.logfun("aa frequency: ")
        self.logfun(count)
        
    def parse_raw_sequence(self, raw_sequence: str):
        raw_sequence_len = len(raw_sequence)
        peptide = []
        index = 0
        while index < raw_sequence_len:
            if raw_sequence[index] == "(":
                if peptide[-1] == "C" and raw_sequence[index:index + 8] == "(+57.02)":
                    index += 8
                else:
                    return False, peptide
            else:
                peptide.append(raw_sequence[index])
                index += 1
        return True, peptide    
        
    def __len__(self):
        return len(self.feature_list)
    
    def close(self):
        self.input_spectrum_handle.close()
        
    def parse_spectrum_ion(self):

        mz_list = []
        intensity_list = []
        line = self.input_spectrum_handle.readline()
        while not "END IONS" in line:
            mz, intensity = re.split(' |\r|\n', line)[:2]
            mz_float = float(mz)
            intensity_float = float(intensity)
            # skip an ion if its mass > MZ_MAX
            if mz_float > configuration.MZ_MAX:
                line = self.input_spectrum_handle.readline()
                continue
            mz_list.append(mz_float)
            intensity_list.append(intensity_float)
            line = self.input_spectrum_handle.readline()
        return mz_list, intensity_list        
        
    def get_feature(self, feature: DDAFeature):
        spectrum_location = self.spectrum_location_dict[feature.scan]
        self.input_spectrum_handle.seek(spectrum_location)
        # parse header lines
        line = self.input_spectrum_handle.readline()
        assert "BEGIN IONS" in line, "Error: wrong input BEGIN IONS"
        line = self.input_spectrum_handle.readline()
        assert "TITLE=" in line, "Error: wrong input TITLE="
        line = self.input_spectrum_handle.readline()
        assert "PEPMASS=" in line, "Error: wrong input PEPMASS="
        line = self.input_spectrum_handle.readline()
        assert "CHARGE=" in line, "Error: wrong input CHARGE="
        line = self.input_spectrum_handle.readline()
        assert "SCANS=" in line, "Error: wrong input SCANS="
        line = self.input_spectrum_handle.readline()
        assert "RTINSECONDS=" in line, "Error: wrong input RTINSECONDS="
        line = self.input_spectrum_handle.readline() 
        assert "SEQ=" in line, "Error: wrong input SEQ="
        mz_list, intensity_list = self.parse_spectrum_ion()
        return get_feature(mz_list, intensity_list, feature, self,self.transform)
        
        
    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = open(self.spectrum_filename,"r")
        feature = self.feature_list[idx]
        return self.get_feature(feature)
    

class SMSDataset(Dataset):
    def __init__(self, spectrum_filename, num, sep, transform = transform_function, vocab = ProteinVocab(), logfun = print, reuse = True):
        self.spectrum_filename      = spectrum_filename
        self.num = num
        self.sep = sep
        self.vocab                  = vocab
        self.input_spectrum_handle  = None
        self.feature_list           = []
        self.spectrum_location_dict = dict()
        self.transform              = transform
        self.logfun                 = logfun
        
        self.knap = KnapsackSearcher(configuration.MZ_MAX,'knapsack.npy', vocab.aa2mass_num)
        

        skipped_by_mass = 0
        skipped_by_ptm = 0
        skipped_by_length = 0
        count = {}
        total_num = 0
        
        for i in  range(num):
            with open(os.path.join(spectrum_filename,f"train_{i}.csv"), "r") as fr:
                line  = True
                while line:
                    scan_index = fr.tell()
                    line = fr.readline()
                    
                    if line == '':
                        continue
                    pep, charge, mz, _, _, specturm = line.strip().split(sep)
                    mass        = (float(float(mz) - configuration.mass_H) * float(charge))
                    ok, pep = self.parse_raw_sequence(pep)
                    
                    if not ok:
                        skipped_by_ptm  += 1
                        continue           
                    if mass > configuration.MZ_MAX:
                        skipped_by_mass += 1
                        continue
                    if len(pep) >= configuration.MAX_LEN:
                        skipped_by_length +=1
                        continue
                    
                    for aa in pep:
                        count[aa] = count.get(aa,0) + 1
                        total_num +=1
                    
                    new_feature = DDAFeature(mz=float(mz),
                                             z=float(charge),
                                             peptide=pep,
                                             scan=f"{i}|{scan_index}",
                                             mass=mass)
                    self.feature_list.append(new_feature)
        self.logfun(f"Total number of PTM: {len(self.feature_list)}, "
              f"Invalid amino acid: {skipped_by_ptm}, "
              f"Large mass: {skipped_by_mass}, "
              f"Long sequency: {skipped_by_length} "
            )
        
        for k,v in count.items():
            count[k] = v/total_num
        self.logfun("aa frequency: ")
        self.logfun(count)        

    def __len__(self):
        return len(self.feature_list)
    
    def parse_raw_sequence(self, raw_sequence:str):
        pep = []
        for r in raw_sequence:
            if r.islower() or r not in configuration.vocab_reserve:
                return False, pep
            pep.append(r)
        return True, pep
            
    def __getitem__(self, idx):
        if self.input_spectrum_handle is None:
            self.input_spectrum_handle = [open(os.path.join(self.spectrum_filename, f"train_{i}.csv"),"r") for i in range(self.num)]
        feature = self.feature_list[idx]
        return self.get_feature(feature)        

    def parse_spectrum_ion(self,file_num):
        line = self.input_spectrum_handle[file_num].readline()
        _, _, _, _, _, specturm = line.strip().split(self.sep)
        specturm = list(map(float,specturm.split(",")))
        mz = specturm[::2]
        intensity = specturm[1::2]
        return mz,intensity

    def get_feature(self, feature: DDAFeature):
        file_num, spectrum_location = feature.scan.split("|") 
        file_num, spectrum_location  = int(file_num), int(spectrum_location )
        self.input_spectrum_handle[file_num].seek(spectrum_location)
        mz_list, intensity_list = self.parse_spectrum_ion(file_num)
        return get_feature(mz_list, intensity_list, feature, self, self.transform) 
      

def collate_func_denovo(data):
    
    batch_max_seq_len = max([len(x[2]) for x in data])
    ion_index_shape = data[0][3][0].shape
    assert ion_index_shape == (23, 12)
    
    peak_location = [x[0] for x in data]
    peak_location = np.stack(peak_location) # [batch_size, N]
    peak_location = torch.from_numpy(peak_location)

    peak_intensity = [x[1] for x in data]
    peak_intensity = np.stack(peak_intensity) # [batch_size, N]
    peak_intensity = torch.from_numpy(peak_intensity)
    
    point_length  = [x[6] for x in data]
    point_length  = torch.LongTensor(point_length)
        
    batch_forward_ion_index = []
    batch_forward_id_target = []
    batch_forward_id_input = []
    ion_index_mask = []
    history_mass   = []
    current_ions = []
    distances = []
    
    for x in data:
        ion_index = np.zeros((batch_max_seq_len, ion_index_shape[0], ion_index_shape[1]),
                               np.float32)
        forward_ion_index = np.stack(x[3])
        ion_index[:forward_ion_index.shape[0], :, :] = forward_ion_index
        batch_forward_ion_index.append(ion_index)
        
        mask = np.array([False] * batch_max_seq_len)
        mask[:forward_ion_index.shape[0]] = True
        ion_index_mask.append(mask)
        
        hmass = np.zeros((batch_max_seq_len,2))
        hmass[:forward_ion_index.shape[0]] = np.array(x[5], np.float32)
        history_mass.append(hmass)

        current_ion_index = np.zeros((batch_max_seq_len,12))
        current_ion_index[:forward_ion_index.shape[0]] = np.array(x[7],np.float32)
        current_ions.append(current_ion_index)

        distance = np.zeros((batch_max_seq_len,22))
        distance[:forward_ion_index.shape[0]] = np.array(x[8],np.float32)
        distances.append(distance)


        f_target = np.zeros((batch_max_seq_len,), np.int64)
        forward_target = np.array(x[2], np.int64)
        f_target[:forward_target.shape[0]] = forward_target
        batch_forward_id_target.append(f_target)

        f_input = np.zeros((batch_max_seq_len,), np.int64)
        forward_input = np.array(x[4], np.int64)
        f_input[:forward_input.shape[0]] = forward_input
        batch_forward_id_input.append(f_input)

    distances = torch.FloatTensor(distances)
    ion_index_mask = torch.BoolTensor(ion_index_mask)
    history_mass   = torch.FloatTensor(history_mass)
    current_ions   = torch.FloatTensor(current_ions)
    batch_forward_id_target = torch.from_numpy(np.stack(batch_forward_id_target))  # [batch_size, T]
    batch_forward_ion_index = torch.from_numpy(np.stack(batch_forward_ion_index))  # [batch, T, 26, 8]
    batch_forward_id_input = torch.from_numpy(np.stack(batch_forward_id_input))
    
    max_point_len = point_length.max()    
    
    return {"mz": peak_location[:,:max_point_len],
            "intensity": peak_intensity[:,:max_point_len],
            "aa_target": batch_forward_id_target,
            "aa_ion" : batch_forward_ion_index,
            "aa_input"   : batch_forward_id_input,
            "aa_mask"  : ion_index_mask,
            "h_mass"   : history_mass,
            "p_len"    : point_length,
            "c_ion"    : current_ions,
            "distance" : distances,
            }
    
    
#from torch.utils.data import DataLoader        
#dataset =  DenovoDataset("/Users/u6169130/Desktop/RAWDATA/Denovo/data/cross.9high_80k.exclude_bacillus/cross.cat.mgf.test.repeat")                
#dataloader = DataLoader(dataset,batch_size=4, shuffle= False, num_workers = 0, collate_fn = collate_func_denovo)                
       
#from tqdm import tqdm

#i = 0         
#for i,data in tqdm(enumerate(dataloader)):
#    pass   
            