import numpy as np
import torchvision
import util
import matplotlib.pyplot as plt
import visual_words
import visual_recog
import deep_recog
import skimage.io

if __name__ == '__main__':
    # num_cores = util.get_num_CPU()

    # path_img = "../data/kitchen/sun_aasmevtpkslccptd.jpg"
    # path_img = "../data/kitchen/sun_axbidhchxcpixhso.jpg"
    # path_img = "../data/waterfall/sun_aastyysdvtnkdcvt.jpg"
    # image = skimage.io.imread(path_img)
    # image = image.astype('float') / 255
    # filter_responses = visual_words.extract_filter_responses(image)
    # util.display_filter_responses(filter_responses)

    # visual_words.compute_dictionary(num_workers=4)


    # dictionary = np.load('../results/dictionary.npy')
    # img = visual_words.get_visual_words(image, dictionary)
    # # visual_recog.get_feature_from_wordmap(img, dictionary.shape[0])
    #
    # plt.imshow(img, cmap='rainbow')
    # plt.savefig('../results/wrong_kitchen.png')
    # plt.show()

    # util.save_wordmap(wordmap, filename)
    # visual_recog.build_recognition_system(num_workers=6)

    # conf, accuracy = visual_recog.evaluate_recognition_system(num_workers=6)
    # print(conf)
    # print(accuracy)
    # wrong = np.load("../results/wrong_prediction.npy")
    # for item in wrong:
    #     file_path, actual, predict = item
    #     print(file_path, actual, predict)
        # if actual == 1 and predict == 7:
        #     print(file_path)

    vgg16 = torchvision.models.vgg16(pretrained=True).double()
    vgg16.eval()
    # deep_recog.build_recognition_system(vgg16,num_workers=num_cores//2)
    # conf = deep_recog.evaluate_recognition_system(vgg16,num_workers=num_cores//2)
    # print(conf)
    # print(np.diag(conf).sum()/conf.sum())
