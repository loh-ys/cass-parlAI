from parlai.core.params import ParlaiParser
from parlai.core.script import ParlaiScript, register_script
import parlai.utils.logging as logging
import parlai.core.opt as op

import random

def setup_args(parser=None):
    if parser is None:
        parser = ParlaiParser(
            True, True, '[WIP] Saving Opt files'
        )
    return parser

def interactive(opt):
    if isinstance(opt, ParlaiParser):
        logging.error('interactive should be passed opt not Parser')
        opt = opt.parse_args()

    # Save opt
    f = input("opt name:")
    opt.save(filename=f)
    print("opt file saved")

@register_script('save_opt', aliases=['so'])
class Interactive(ParlaiScript):
    @classmethod
    def setup_args(cls):
        return setup_args()

    def run(self):
        return interactive(self.opt)


if __name__ == '__main__':
    random.seed(42)
    Interactive.main()