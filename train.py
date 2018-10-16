import model.model_base as mb

if __name__ == '__main__':
    mb.init()
    mb.fw.train(mb.model, ckpt_dir="checkpoint",
                model_name=mb.dataset_name + "_" + mb.model.encoder + "_" + mb.model.selector,
                max_epoch=60, gpu_nums=1)
