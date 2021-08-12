from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import parlai.core.opt as op
from parlai.core.agents import create_agent

import random

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, '[WIP] Loading Opt files'
        )
    return parser

def interactive(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()
    models = []
    labels = []
    m = input("how many models:")
    m = int(m)

    # Loop to load opt
    i = 1
    while i <= m:
        f = input("opt filename to load:")
        opt = op.Opt.load(f)
        print("opt file loaded")
        models.append(create_agent(opt))

        i += 1

@register_script('load_opt', aliases=['lo'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()