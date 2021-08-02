from parlai.core.params import ParlaiParser
from parlai.core.agents import create_agent
from parlai.core.worlds import create_task
from parlai.core.script import ParlaiScript, register_script
from parlai.utils.world_logging import WorldLogger
from parlai.agents.local_human.local_human import LocalHumanAgent
import parlai.utils.logging as logging


from parlai.core.build_data import modelzoo_path
from parlai.core.params import get_model_name

import random
import os


# note: assume that ParlAI has been installed into home
path = os.path.expanduser('~/ParlAI')
print(path)
os.chdir(path)


class CassQuipWorld():

    human_agent = None
    models = None
    models_labels = None
    
    def __init__(self, human_agent, models, models_labels):
        self.human_agent = human_agent
        self.models = models
        self.models_labels = models_labels


    def parley(self):

        human_input = self.human_agent.act()
        suggestions = []
                    
        for model in self.models:
            model.observe(human_input)
            suggestion = model.act()
            suggestions.append(suggestion['text'])

        print("CassQuips suggestions:")
        for i in range(len(self.models)):
            print(self.models_labels[i] + ": " +  suggestions[i])           
            
    


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

    #get path to Convai, Blended skill talk, and Empathetic dialogue model files

    cv_model =  modelzoo_path(opt['datapath'], cv_zoo)
    ed_model =  modelzoo_path(opt['datapath'], ed_zoo)
    bst_model = modelzoo_path(opt['datapath'], bst_zoo)
    
    print(cv_model)
    print(ed_model)
    print(bst_model)

    # update convai options
    # args to change: download_path, datapath, model_file, parlai_home, override
    over_convai = {
                   'download_path': opt['download_path'],
                   'datapath': opt['datapath'],
                   'model_file': cv_model,
                   'parlai_home': opt['parlai_home'],
                   'override': {'model_file': cv_model,
                                 'task': 'convai2'}
                   }   
    
    #updates to empathetic dialogue options
    over_ed = {
               'download_path': opt['download_path'],
               'datapath': opt['datapath'],
               'model_file': ed_model,
               'parlai_home': opt['parlai_home'],
               'override': {'model_file': ed_model}
               
               }   
    # updates to blended skill talk options
    over_bst = {
                'download_path': opt['download_path'],
                'datapath': opt['datapath'],
                'model_file': bst_model,
                'parlai_home': opt['parlai_home'],
                'override': {'model_file': bst_model,
                             'task': 'blended_skill_talk'}
                
                }   

    #update opt dictionaries

    opt_convai.update(over_convai)
    opt_ed.update(over_ed)
    opt_bst.update(over_bst)

    

    human_agent = LocalHumanAgent(opt_convai)
        
    convai_agent = create_agent(opt_convai, requireModelExists=True)
    ed_agent = create_agent(opt_ed, requireModelExists=True)
    bst_agent = create_agent(opt_bst, requireModelExists=True)

    models = [convai_agent, ed_agent, bst_agent]
    labels = ['CONV AI', 'EMPATHETIC DIALOGUE', 'Blended Skill Talk']
    
    cass_quips = CassQuipWorld(human_agent, models, labels)

    keep_suggesting = True
    while(keep_suggesting):
        cass_quips.parley()
        user_input = input("")
        if user_input == "EXIT":
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



