# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 14:00:05 2019

@author: Marti
"""

from more_itertools import powerset
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
from ipywidgets import  widgets

class GameMaster():
    
    import json
    from scipy import spatial
    import torch
    import csv
    import time
    import numpy as np
    from random import sample     
    from gensim.models.word2vec import Word2Vec
    import gensim.downloader as api
    import copy    
    
    
    def __init__(self):
        self._model_gigaword = self.api.load("glove-wiki-gigaword-100")
        self._vocab = self._load_vocab()
        self._cards = self._import_words()
        self._board_state = self._initiate_board_state(self._cards)
        self._all_sims = self._create_all_similarities(self._board_state)
        self._board = [self._board_state['words'][i:i+5] for i in range(0, len(self._board_state['words']), 5)]
        self.player = 'Red'
        
        
    def _load_vocab(self):
        vocab_path = 'data/my_vocab_GENSIM.json'
        with open(vocab_path) as json_file:
            loaded_vocab = self.json.load(json_file)
    
        for key in loaded_vocab.keys():
            loaded_vocab[key] = [float(x) for x in loaded_vocab[key]]
        return loaded_vocab
    
    
    def _import_words(self):
        word_path = 'data/words.txt'
        f = open(word_path, "r")
        temp_words = f.readlines()
        temp_words = [word.lower().replace('\n','') for word in temp_words]
        temp_words = list(set(temp_words))
        words = []
        
        # Only keep words that are in the vocabulary
        for word in temp_words: 
            if word in self._vocab.keys(): 
                words.append(word)
                
        return words
    
    
    def change_player(self):
        if self.player == 'Red':
            self.player = 'Blue'
        elif self.player == 'Blue':
            self.player = 'Red'
    
    
    def _initiate_board_state(self, words):
        chosen_words = self.sample(words,25)
        words_for_sample = chosen_words
        red = self.sample(words_for_sample,9)
        words_for_sample = list(set(words_for_sample).difference(set(red)))
        blue = self.sample(words_for_sample,8)
        words_for_sample = list(set(words_for_sample).difference(set(blue)))
        civilians = self.sample(words_for_sample,7)
        words_for_sample = list(set(words_for_sample).difference(set(civilians)))
        assassin = words_for_sample
        
        board_state = {'words':chosen_words, 
                       'Red':red,'Red_killed':[],'Red_hints':[], 
                       'Blue':blue,'Blue_killed':[], 'Blue_hints':[], 
                       'civilians':civilians,'civilians_killed':[], 
                       'assassin':assassin,'assassin_killed':[]}
     
        return board_state
    
    
    def reset_board_state(self):
        self._board_state = self._initiate_board_state(self._cards)
        self._all_sims = self._create_all_similarities(self._board_state)     
        self._board = [self._board_state['words'][i:i+5] for i in range(0, len(self._board_state['words']), 5)]
        self.player = 'Red'
    
    
    def _create_all_similarities(self, board):
        
        all_red_sims = {}
        all_blue_sims = {}
        all_civilians_sims = {}
        assassin_sims = {}
    
        for spy in board['Red']:
            sim = self._create_similarities(spy)
            all_red_sims[spy] = sim
    
        for spy in board['Blue']:
            sim = self._create_similarities(spy)
            all_blue_sims[spy] = sim
    
        for spy in board['civilians']:
            sim = self._create_similarities(spy)
            all_civilians_sims[spy] = sim
    
        for spy in board['assassin']:
            sim = self._create_similarities(spy)
            assassin_sims[spy] = sim
        
        all_sims = {'Red':all_red_sims,'Blue':all_blue_sims,'civilians':all_civilians_sims,'assassin':assassin_sims} 
        
        return all_sims
    
    
    def _create_similarities(self, spy_word):

        similarities = self._model_gigaword.most_similar(spy_word, topn=10000)
            
        sim_dict = self._create_sim_dict(similarities)  
        cleaned_sim = self._clean_similarities(self._board_state['words'], sim_dict)
            
        return cleaned_sim
    
    
    def _clean_similarities(self, board_words, sim):
        cleaned_sim = sim
        guesses = list(sim.keys())
        for board_word in board_words:
            for guess in guesses:
                # Delete the guess if it is a sub-string of a word on the board
                if guess in board_word or board_word in guess:
                    try:
                        del cleaned_sim[guess]
                    except:
                        continue
        return cleaned_sim
        
    
    def _create_sim_dict(self, sim_list):
        sim_dict = {}
        for entry in sim_list:
            sim_dict[entry[0]] = entry[1]
        return sim_dict
    
    
    def _create_word_groups(self, sub_board):
        ps = list(powerset(sub_board))
        candidates = []
        for word_group in ps:
            if len(word_group) > 0 and len(word_group) <= 3:
                 candidates.append(list(word_group))
            if len(word_group) > 3: # No need to iterate through all combinations
                break
        return candidates
    
    
    def _compute_hint(self, similarities):
        combined_similarity = {}
        for key in similarities[0].keys():
            try: # Will fail if the word is only in one of the similarity dicts
                avg_score = self.np.mean([x[key] for x in similarities])
                if len(similarities) == 2:
                    avg_score += 0.25
                if len(similarities) == 3:
                    avg_score += 0.28
                combined_similarity[key] = avg_score
            except:
                continue
            
        ranked_similiarity = [(key,combined_similarity[key]) for key in combined_similarity.keys()]
        ranked_similiarity = sorted(ranked_similiarity, key=lambda x: x[1])
        
        if len(ranked_similiarity) > 0:
            return ranked_similiarity[-1]
        else:
            return (key, .0)
        
        
    def final_hint(self):
        word_groups = self._create_word_groups(self._board_state[self.player])
        current_sims = self._all_sims[self.player]
        hints = []
        for wg in word_groups:
            sims = [current_sims[word] for word in wg]
            hints.append((wg, self._compute_hint(sims)))   

        ranked_hints = sorted(hints, key=lambda x: x[1][1])
        ranked_hints.reverse()
        for hint in ranked_hints:
            hint_passed = self._evaluate_hint(hint[1][0], self._all_sims)
            if hint_passed:
                self._board_state['{}_hints'.format(self.player)].append(hint[1][0])
                return hint[1][0], len(hint[0])
                #return hint
            
        
    def _evaluate_hint(self, hint, all_sims):
        if hint in self._board_state['{}_hints'.format(self.player)]:
            return False
               
        for assassin in all_sims['assassin'].keys():
            try:
                assass_score = all_sims['assassin'][assassin][hint]
            except:
                assass_score = 0
            if assass_score < 0.5:
                continue
            else:
                print ('hint failed, assassin')
                return False
    
        if self.player == 'Red':
            evaluate_sim = all_sims['Blue']
        elif self.player == 'Blue':
            evaluate_sim = all_sims['Red']
            
        for spy in evaluate_sim.keys():
            try:
                eval_score = evaluate_sim[spy][hint]
            except:
                eval_score= 0
            if eval_score < 0.5:
                continue
            else:
                print ('hint failed, other player')
                return False
        
        for civ in all_sims['civilians'].keys():
            try:
                civ_score = all_sims['assassin'][civ][hint]
            except:
                civ_score = 0
            if civ_score < 0.5:
                continue
            else:
                print ('hint failed, civilian')
                return False
            
        return True
    
    
    def kill(self, target):
        target_buckets = ['Red', 'Blue', 'civilians', 'assassin']
        for bucket in target_buckets:
            for spy in self._board_state[bucket]:
                if spy == target:
                    self._board_state[bucket].remove(target)
                    self._board_state['{}_killed'.format(bucket)].append(target)
                    del self._all_sims[bucket][target]


    def _get_board_colors(self):
        colors =  [["w","w","w","w","w"],
                   ["w","w","w","w","w"],
                   ["w","w","w","w","w"],
                   ["w","w","w","w","w"],
                   ["w","w","w","w","w"]]
        for word in self._board_state['Red_killed']:
            pos = self._get_position(word)
            colors[pos[0]][pos[1]] = 'red'
        for word in self._board_state['Blue_killed']:
            pos = self._get_position(word)
            colors[pos[0]][pos[1]] = 'blue'
        for word in self._board_state['civilians_killed']:
            pos = self._get_position(word)
            colors[pos[0]][pos[1]] = 'grey'
        for word in self._board_state['assassin_killed']:
            pos = self._get_position(word)
            colors[pos[0]][pos[1]] = 'purple'
        return colors
    
    def _get_position(self, word):
        pos = [(i, colour.index(word))
               for i, colour in enumerate(self._board)
               if word in colour]
        return pos[0]


    def _et_button_clicked(self, b):
        self.change_player()
        print('{} players turn'.format(self.player))


    def _kill_button_clicked(self, b):
        self.kill(self._selected_target)
        self.plot_board()


    def _hint_button_clicked(self, b):
        fh = self.final_hint()
        #print(fh)
        print('{}, {}'.format(str(fh[0].upper()), str(fh[1])))


    def _selection_change(self, change):
        if change['type'] == 'change' and change['name'] == 'value':
            self._selected_target = change['new']


    def produce_hint(self):
        hint_button = widgets.Button(description='Hint')
        output = widgets.Output()
        display(hint_button, output)
        hint_button.on_click(self._hint_button_clicked)        


    def plot_board(self):
        clear_output()
        options = self._board_state['words']
        options.sort()
        spy_selector =  widgets.Dropdown(
            options = options,
            description='Select Target',
            disabled=False,
            value=None
            )
        display(spy_selector)
        spy_selector.observe(self._selection_change)

        kill_button = widgets.Button(description='Eliminate')
        output = widgets.Output()
        display(kill_button, output)
        kill_button.on_click(self._kill_button_clicked)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.axis('tight')
        ax.axis('off')
        table = ax.table(cellText=self._board,loc='center',cellLoc='center',
                         cellColours=self._get_board_colors())
        table.set_fontsize(14)
        table.scale(1.5, 3.5)
        plt.show()
                    
        et_button = widgets.Button(description='End Turn')
        output = widgets.Output()
        display(et_button, output)
        
        et_button.on_click(self._et_button_clicked)
        