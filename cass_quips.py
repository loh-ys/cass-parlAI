import random
from pathlib import Path
import os

import pandas as pd

from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging


from parlai.core.build_data import modelzoo_path
from parlai.core.params import get_model_name



# note: assume that ParlAI has been installed into home
# changing the working directory is necessary to create the ConvAi agent
path = os.path.expanduser('~/ParlAI')
print(path)
os.chdir(path)


class CassQuipWorld():

    human_agent = None
    models = None
    model_labels = None
    suggestions = []
    
    # path in which to store chat logs
    default_path = os.path.expanduser('~/ParlAI/cass_quips')

    def __init__(self, human_agent, models, model_labels, 
                 response = None, log_path = default_path):
        self.human_agent = human_agent
        self.models = models
        self.model_labels = model_labels
        self.response = response
        self.log_path = log_path
    
        
        # initialise empty dataframe to store chat logs
        self.out_log = pd.DataFrame(columns = ['choice', 'text'])

    
    def log(self):
        '''
        Function to store a history of the chat. Stores the selected response and the text of the response.
        '''
        #get the choice of model and response text 
        log = {'choice': [self.response['choice']],
                   'text': [self.response['text']]}    

        #print(log)

        # append this to existing dataframe
        self.out_log = self.out_log.append(log, ignore_index = True)        
        
        #print(log)

        


    def parley(self):
        '''
        Parley function used in the ParlAI framework. This function passes an observation dict to the three models and stores 
        their responses.

        Upon initialisation, this observation dict is generated by observing human input from a LocalHumanAgent. In subsequent
        rounds, the act dict from the response chosen in the previous round is observed.
        '''
        self.suggestions = []
        self.suggest_dict = []

        if self.response is None:
            human_input = self.human_agent.act()
                        
            for model in self.models:
                model.observe(human_input)
                #print(model.observe(human_input))
                suggestion = model.act()
                #print(suggestion)
                self.suggestions.append(suggestion['text'])
                self.suggest_dict.append(suggestion)

            print("CassQuips suggestions:")
            for i in range(len(self.models)):
                print(self.model_labels[i] + ": " +  self.suggestions[i])    

            #Log the initial input passed to the models.
            self.response = {'choice': 'initial',
                             'text': human_input['text']}
            self.log()

        else:
            input = self.response

            for model in self.models:
                model.observe(input)
                #print(model.observe(input))
                suggestion = model.act()
                self.suggestions.append(suggestion['text'])
                self.suggest_dict.append(suggestion)

            print("CassQuips suggestions:")
            for i in range(len(self.models)):
                print(self.model_labels[i] + ": " +  self.suggestions[i])    

    def preference(self):       
        '''
        Function that allows users to choose a preferred response, or generate a personalised response.
        These choices are also logged to improve future model performance.
        '''
        self.response = {}

        print('Select a preferred response or input a personalised response')
        print(f"""
                Type 'A' for {self.model_labels[0]},
                Type 'B' for {self.model_labels[1]},
                Type 'C' for {self.model_labels[2]},
                Type 'D' to enter your own input
                """)

        user_input = input()

        if user_input == 'A':
            # get text of the response from the observation dictionary and store
            response = self.suggest_dict[0]            

            # store the observation dictionary, and add the model chosen
            self.response = response
            self.response['choice'] = self.model_labels[0]

            print(f'Response: {self.response["text"]}')

            #log the choices
            print('Logging choices...')
            self.log()
            

        elif user_input == 'B':
            response = self.suggest_dict[1]

            self.response = response
            self.response['choice'] = self.model_labels[1]

            # print responses
            print(f'Response: {self.response["text"]}')

            print('Logging choices...')
            self.log()
        
        elif user_input == 'C':
            response = self.suggest_dict[2]

            self.response = response
            self.response['choice'] = self.model_labels[2]

            print(f'Response: {self.response["text"]}')
            
            print('Logging choices...')
            self.log()

        elif user_input == 'D':
            print('Please enter your reply:')
            response = self.human_agent.act()

            self.response = response
            self.response['choice'] = 'own_input'

            print(f'Response: {self.response["text"]}')

            print('Logging choices...')
            self.log()            

        else:
            print('Invalid input; please only enter A, B, C or D')
            self.preference()

        
            
    

# convai
# python parlai/scripts/interactive.py -mf zoo:pretrained_transformers/model_poly/model -t convai2
opt_convai = {'init_opt': None, 'task': 'convai2', 'download_path': '/home/ahmad/ParlAI/downloads', 'loglevel': 'success', 
              'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 
              'dynamic_batching': None, 'datapath': '/home/ahmad/ParlAI/data', 'model': None, 
              'model_file': '/home/ahmad/ParlAI/data/models/pretrained_transformers/model_poly/model', 'init_model': None, 
              'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 
              'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 
              'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 
              'image_size': 256, 'image_cropsize': 224, 'interactive_mode': True, 'embedding_type': 'random', 
              'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 
              'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 
              'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 
              'truncate': 1024, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 
              'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 
              'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'lr_scheduler': 'reduceonplateau', 
              'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 
              'warmup_rate': 0.0001, 'update_freq': 1, 'candidates': 'inline', 'eval_candidates': 'inline', 
              'interactive_candidates': 'fixed', 'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 
              'fixed_candidate_vecs': 'reuse', 'encode_candidate_vecs': True, 'encode_candidate_vecs_batchsize': 256, 
              'train_predict': False, 'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'inference': 'max', 
              'topk': 5, 'return_cand_scores': False, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 
              'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 
              'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'n_encoder_layers': -1, 
              'n_decoder_layers': -1, 'model_parallel': False, 'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 
              'normalize_sent_emb': False, 'share_encoders': True, 'share_word_embeddings': True, 'learn_embeddings': True, 'data_parallel': False, 
              'reduction_type': 'mean', 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 
              'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 
              'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 
              'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'polyencoder_type': 'codes', 
              'poly_n_codes': 64, 'poly_attention_type': 'basic', 'poly_attention_num_heads': 4, 'codes_attention_type': 'basic', 
              'codes_attention_num_heads': 4, 'display_partner_persona': True, 'parlai_home': '/home/ahmad/ParlAI', 
              'override': {'model_file': '/home/ahmad/ParlAI/data/models/pretrained_transformers/model_poly/model', 'task': 'convai2'}, ## CHANGE MODEL PATH
              'starttime': 'Aug24_12-51'}

# ed
#python parlai interactive -mf zoo:dodecadialogue/empathetic_dialogues_ft/model
opt_ed = {'init_opt': None, 'task': 'interactive', 'download_path': '/home/ahmad/ParlAI/downloads', 'loglevel': 'info', 
          'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 
          'dynamic_batching': None, 'datapath': '/home/ahmad/ParlAI/data', 'model': None, 
          'model_file': '/home/ahmad/ParlAI/data/models/dodecadialogue/empathetic_dialogues_ft/model', ## CHANGE MODEL FILE PATH
          'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 
          'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 
          'save_format': 'conversations', 'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 
          'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 
          'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 
          'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 
          'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 
          'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 'beam_length_penalty': 0.65, 
          'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 
          'temperature': 1.0, 'compute_tokenized_bleu': False, 'interactive_mode': True, 'embedding_type': 'random', 
          'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 'force_fp16_tokens': False, 'optimizer': 'sgd', 
          'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 
          'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 
          'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 
          'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 
          'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 
          'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 
          'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 
          'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 
          'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 
          'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 
          'update_freq': 1, 'image_features_dim': 2048, 'image_encoder_num_layers': 1, 'n_image_tokens': 1, 'n_image_channels': 1, 
          'include_image_token': True, 'image_fusion_type': 'late', 'parlai_home': '/home/ahmad/ParlAI', 
          'override': {'model_file': '/home/ahmad/ParlAI/data/models/dodecadialogue/empathetic_dialogues_ft/model'}, 'starttime': 'Aug24_14-42'}

# bst
# python parlai interactive -mf zoo:blended_skill_talk/bst_single_task/model -t blended_skill_talk
opt_bst = {'init_opt': None, 'task': 'blended_skill_talk', 'download_path': '/home/ahmad/ParlAI/downloads', 'loglevel': 'info', 
            'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 
            'datapath': '/home/ahmad/ParlAI/data', 'model': None, 
            'model_file': '/home/ahmad/ParlAI/data/models/blended_skill_talk/bst_single_task/model', 
            'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 
            'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 
            'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 
            'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'apex', 
            'force_fp16_tokens': False, 'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 
            'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 
            'rank_candidates': False, 'truncate': 1024, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 
            'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 
            'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 
            'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'max_lr_steps': -1, 
            'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'candidates': 'inline', 
            'eval_candidates': 'inline', 'interactive_candidates': 'fixed', 'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 
            'fixed_candidate_vecs': 'reuse', 'encode_candidate_vecs': True, 'encode_candidate_vecs_batchsize': 256, 'train_predict': False, 
            'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'inference': 'max', 'topk': 5, 
            'return_cand_scores': False, 'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 
            'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 
            'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 
            'model_parallel': False, 'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False, 
            'share_encoders': True, 'share_word_embeddings': True, 'learn_embeddings': True, 'data_parallel': False, 'reduction_type': 'mean', 
            'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 
            'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 
            'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 
            'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'hf_skip_special_tokens': True, 'polyencoder_type': 'codes', 
            'poly_n_codes': 64, 'poly_attention_type': 'basic', 'poly_attention_num_heads': 4, 'codes_attention_type': 'basic', 
            'codes_attention_num_heads': 4, 'display_partner_persona': True, 'include_personas': True, 'include_initial_utterances': False, 
            'safe_personas_only': True, 'parlai_home': '/home/ahmad/ParlAI', 
            'override': {'model_file': '/home/ahmad/ParlAI/data/models/blended_skill_talk/bst_single_task/model', 'task': 'blended_skill_talk'}, 
            'starttime': 'Aug24_13-01'}
# eli5 
#parlai interactive -mf zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent
opt_eli5 = {'init_opt': None, 'allow_missing_init_opts': False, 'task': 'interactive', 'download_path': None, 
            'loglevel': 'info', 'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 
            'batchsize': 1, 'dynamic_batching': None, 'verbose': False, 'is_debug': False, 'datapath': '/home/dell/ParlAI/data', 
            'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent', 
            'model_file': '/home/dell/ParlAI/data/models/dialogue_unlikelihood/rep_eli5_ctxt_and_label/model', 
            'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 
            'display_add_fields': '', 'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 
            'local_human_candidates_file': None, 'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 
            'embedding_size': 300, 'n_layers': 2, 'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 
            'learn_positional_embeddings': False, 'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 
            'activation': 'relu', 'output_scaling': 1.0, 'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 
            'model_parallel': False, 'beam_size': 1, 'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 
            'beam_block_full_context': True, 'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 
            'topp': 0.9, 'beam_delay': 30, 'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False, 
            'interactive_mode': True, 'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'safe', 
            'force_fp16_tokens': False, 'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 
            'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 
            'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 'label_truncate': None, 'history_reversed': False, 'history_size': -1, 
            'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False, 'delimiter': '\n', 
            'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 
            'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 
            'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 
            'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 'bpe_dropout': None, 
            'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'invsqrt_lr_decay_gamma': -1, 'warmup_updates': -1, 
            'warmup_rate': 0.0001, 'update_freq': 1, 'image_features_dim': 2048, 'image_encoder_num_layers': 1, 'n_image_tokens': 1, 'n_image_channels': 1, 
            'include_image_token': True, 'image_fusion_type': 'late', 'seq_ul_ratio': 0.5, 'seq_ul_n': 4, 'mask_n': 100, 'ctxt_beta': 0.5, 
            'crep_pen': 'crep', 'parlai_home': '/home/dell/ParlAI', 
            'override': {'model_file': '/home/dell/ParlAI/data/models/dialogue_unlikelihood/rep_eli5_ctxt_and_label/model', 'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent'}, 
            'starttime': 'Aug06_01-41'}

# wizard of wikipedia
# parlai interactive -mf zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent
opt_wow = {'init_opt': None, 'allow_missing_init_opts': False, 'task': 'interactive', 'download_path': None, 'loglevel': 'info', 
           'datatype': 'train', 'image_mode': 'raw', 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1, 'dynamic_batching': None, 
           'verbose': False, 'is_debug': False, 'datapath': '/home/dell/ParlAI/data', 
           'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent', 
           'model_file': '/home/dell/ParlAI/data/models/dialogue_unlikelihood/rep_wiki_ctxt_and_label/model', 'init_model': None, 
           'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False, 'display_add_fields': '', 
           'interactive_task': True, 'outfile': '', 'save_format': 'conversations', 'local_human_candidates_file': None, 
           'single_turn': False, 'log_keep_fields': 'all', 'image_size': 256, 'image_cropsize': 224, 'embedding_size': 300, 'n_layers': 2, 
           'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False, 
           'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0, 
           'share_word_embeddings': True, 'n_encoder_layers': -1, 'n_decoder_layers': -1, 'model_parallel': False, 'beam_size': 1, 
           'beam_min_length': 1, 'beam_context_block_ngram': -1, 'beam_block_ngram': -1, 'beam_block_full_context': True, 
           'beam_length_penalty': 0.65, 'skip_generation': False, 'inference': 'greedy', 'topk': 10, 'topp': 0.9, 'beam_delay': 30, 
           'beam_block_list_filename': None, 'temperature': 1.0, 'compute_tokenized_bleu': False, 'interactive_mode': True, 
           'embedding_type': 'random', 'embedding_projection': 'random', 'fp16': False, 'fp16_impl': 'safe', 'force_fp16_tokens': False, 
           'optimizer': 'sgd', 'learningrate': 1, 'gradient_clip': 0.1, 'adam_eps': 1e-08, 'adafactor_eps': (1e-30, 0.001), 'momentum': 0, 
           'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'rank_candidates': False, 'truncate': -1, 'text_truncate': None, 
           'label_truncate': None, 'history_reversed': False, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 
           'add_p1_after_newln': False, 'delimiter': '\n', 'history_add_global_end_token': None, 'special_tok_lst': None, 'gpu': -1, 'no_cuda': False, 
           'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1, 'dict_minfreq': 0, 'dict_maxtokens': -1, 
           'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__', 'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 
           'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'bpe_vocab': None, 'bpe_merge': None, 'bpe_add_prefix_space': None, 
           'bpe_dropout': None, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3, 'lr_scheduler_decay': 0.5, 'invsqrt_lr_decay_gamma': -1, 
           'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'image_features_dim': 2048, 'image_encoder_num_layers': 1, 'n_image_tokens': 1, 
           'n_image_channels': 1, 'include_image_token': True, 'image_fusion_type': 'late', 'seq_ul_ratio': 0.5, 'seq_ul_n': 4, 'mask_n': 100, 
           'ctxt_beta': 0.5, 'crep_pen': 'crep', 'parlai_home': '/home/dell/ParlAI', 
           'override': {'model_file': '/home/dell/ParlAI/data/models/dialogue_unlikelihood/rep_wiki_ctxt_and_label/model', 
           'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent'}, 'starttime': 'Aug06_03-29'}




#python parlai/scripts/interactive.py -mf zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model -m projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent




def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, 'Interactive chat with a model on the command line'
        )
    parser.add_argument('-d', '--display-examples', type='bool', default=False)
    parser.set_defaults(interactive_mode=True, task='interactive')
    LocalHumanAgent.add_cmdline_args(parser)
    WorldLogger.add_cmdline_args(parser)
    return parser


def choose_models():
    '''
    Function for users to choose *up to* 3 models from 5 available to converse with
    '''
    #list of available models
    avail_models = ['convai', 'ed', 'bst', 'eli5', 'wow']

    model_params = {        
        
        'convai': {
                    'opt': opt_convai,
                    'agent': 'convai_agent',
                    'label': 'Convai'
                    },

        'ed':{
                    'opt': opt_ed,
                    'agent': 'ed_agent',
                    'label': 'Empathetic Dialogues'
                },

        'bst': { 
                    'opt': opt_bst,
                    'agent': 'bst_agent',
                    'label': 'Blended Skill Talk'
                },
        
        'eli5': {
                    'opt': opt_eli5,
                    'agent': 'eli5_agent',
                    'label': 'ELI5'
                },
        
        'wow': {
                    'opt': opt_wow,
                    'agent': 'wow_agent',
                    'label': 'Wizard of Wikipedia'
                } 
    }

    print('Available models:')
    labels = [model_params[model]['label'] for model in model_params.keys()]
    for i,v in enumerate(labels, 1):
        print(i,v)


    print(f'''

    To choose a model, enter the number associated with the model. 
    For instance:
    1 to choose {labels[0]}
    2 to choose {labels[1]}, etc.

    To choose multiple models, enter their numbers sequentially
    For instance:
    To choose models 1,2 and 4: input 124
    ''')
    choice = input('Please choose up to three models to use')

    if len(choice) == 0:
        print('Please select at least one model')
        choose_models()

    elif len(choice) > 3:
        print('Please only select up to 3 models')
        choose_models()
    else:
        try:
            selected_options = [int(i) for i in choice]
            #options begin at 1, but python uses 0-based indexing
            model_index = [(i-1) for i in selected_options] 

            #return values if the indices are valid, 
            #throw exception otherwise
            if max(selected_options) <= len(avail_models):
                # use model indices to extract relevant models
                selected_models = [avail_models[i] for i in model_index]

                #get parameters of these models to return
                out_params = [model_params[model] for model in selected_models]
                return(out_params)
            
            else:
                print('One of the chosen models does not exist')
                choose_models()

        except ValueError:
            print('Please only input numbers')
            choose_models()
        




        






def interactive(opt):

    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Create model and assign it to the specified task

    # options reference local files which would break if left in
    # update these options using the paths defined in the default options

    #get paths to model files

    #references to models in the model zoo
    cv_zoo =  'zoo:pretrained_transformers/model_poly/model'
    ed_zoo = 'zoo:dodecadialogue/empathetic_dialogues_ft/model'
    bst_zoo =  'zoo:blended_skill_talk/bst_single_task/model'
    eli5_zoo = 'zoo:dialogue_unlikelihood/rep_eli5_ctxt_and_label/model'
    wow_zoo = 'zoo:dialogue_unlikelihood/rep_wiki_ctxt_and_label/model'

    #get path to Convai, Blended skill talk, and Empathetic dialogue model files

    cv_model =  modelzoo_path(opt['datapath'], cv_zoo)
    ed_model =  modelzoo_path(opt['datapath'], ed_zoo)
    bst_model = modelzoo_path(opt['datapath'], bst_zoo)
    eli5_model= modelzoo_path(opt['datapath'], eli5_zoo)
    wow_model = modelzoo_path(opt['datapath'], wow_zoo)
    
    

    # update convai overrides
    #args to change: download_path, datapath, model_file, parlai_home
    over_convai = {
                   'download_path': opt['download_path'],
                   'datapath': opt['datapath'],
                   'model_file': cv_model,
                   'parlai_home': opt['parlai_home'],
                   'override': {'model_file': cv_model,
                                 'task': 'convai2'}
                   }   
    
    over_ed = {
               'download_path': opt['download_path'],
               'datapath': opt['datapath'],
               'model_file': ed_model,
               'parlai_home': opt['parlai_home'],
               'override': {'model_file': ed_model}
               
               }   

    over_bst = {
                'download_path': opt['download_path'],
                'datapath': opt['datapath'],
                'model_file': bst_model,
                'parlai_home': opt['parlai_home'],
                'override': {'model_file': bst_model,
                             'task': 'blended_skill_talk'}
                
                }   

    
    over_eli5 = {
                'download_path': opt['download_path'],
                'datapath': opt['datapath'],
                'model_file': eli5_model,
                'parlai_home': opt['parlai_home'],
                'override': {'model_file': eli5_model,
                             'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent'}
                
                }   

    over_wow = {
                'download_path': opt['download_path'],
                'datapath': opt['datapath'],
                'model_file': wow_model,
                'parlai_home': opt['parlai_home'],
                'override': {'model_file': wow_model,
                             'model': 'projects.dialogue_unlikelihood.agents:RepetitionUnlikelihoodAgent'}                
                } 

    #update opt dictionaries

    opt_convai.update(over_convai)
    opt_ed.update(over_ed)
    opt_bst.update(over_bst)
    opt_eli5.update(over_eli5)
    opt_wow.update(over_wow)

    # allow users to choose which models to interact with

    select_model_params = choose_models()    

    human_agent = LocalHumanAgent(opt_convai)

    models = [create_agent(model['opt'], requireModelExists=True)
              for model in select_model_params]
        
    # convai_agent = create_agent(opt_convai, requireModelExists=True)
    # ed_agent = create_agent(opt_ed, requireModelExists=True)
    # bst_agent = create_agent(opt_bst, requireModelExists=True)

    #models = [convai_agent, ed_agent, bst_agent]

    labels = [model['label'] for model in select_model_params]
    

    cass_quips = CassQuipWorld(human_agent, models, labels)

    keep_suggesting = True
    while(keep_suggesting):
        cass_quips.parley()
        cass_quips.preference()

        print("Press Enter to continue, type EXIT to stop and record chat logs")
        user_input = input("")
        if user_input == "EXIT":
            #create cass_folder directory in ParlAi directory
            Path(cass_quips.log_path).mkdir(exist_ok= True)

            out_path = cass_quips.log_path + '/recent_chat.csv'

            print(f'Exporting chat logs to: {out_path}')
            cass_quips.out_log.to_csv(out_path, mode = 'a', header = False, index = False) #a mode: appends to file if it exists
            print('Shutting down...')

            keep_suggesting = False
    

    

 
@register_script('cass_quips', aliases=['cq'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()
        

    def run(self):
        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()


