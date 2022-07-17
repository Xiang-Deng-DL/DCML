from config import get_config
from learner import metric_learner
from myutils import set_seed

if __name__ == '__main__':

    conf = get_config()
    
    set_seed(conf.seed)
    
    print(conf)

    learner = metric_learner(conf)

    learner.load_bninception_pretrained(conf)

    learner.train(conf)