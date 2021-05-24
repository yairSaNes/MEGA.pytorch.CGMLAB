import os
from demo.odevalInference import run_inference


def run_inference_vid():

    os.chdir('..')  # mega root

    config_base_file = 'configs/BASE_RCNN_1gpu.yaml'
    # method = 'base'
    method = 'fgfa'
    config_file = 'configs/FGFA/vid_R_50_C4_FGFA_1x_3_frames.yaml'  # config file
    checkpoint_file = 'models/FGFA_R_50.pth'
    input_images_folder = 'datasets/ILSVRC2015_examples/ILSVRC2015_sample3/DATA/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00078000/'
    output_sfx = '_ILSVRC2015_train_00078000'
    # input_images_folder = 'datasets/ILSVRC2015_examples/ILSVRC2015_sample1/DATA/VID/train/ILSVRC2015_VID_train_0000/ILSVRC2015_train_00071005/'
    # output_sfx = '_ILSVRC2015_train_00071005'
    confidence_threshold = 0.7  #
    output_folder = None
    file_suffix = '.jpg'
    # max_num_frames = -1
    max_num_frames = 80

    output_sfx = '{}_confidence_{}'.format(output_sfx, confidence_threshold)

    run_inference(config_base_file,
                  method,
                  config_file,
                  checkpoint_file,
                  input_images_folder,
                  output_sfx,
                  confidence_threshold,
                  output_folder=output_folder,
                  file_suffix=file_suffix,
                  max_num_frames=max_num_frames,
                  )

    pass



if __name__ == '__main__':

    run_inference_vid()

    print('Done!')

