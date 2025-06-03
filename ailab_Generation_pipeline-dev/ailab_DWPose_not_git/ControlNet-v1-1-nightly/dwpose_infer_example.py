from annotator.dwpose import DWposeDetector
import pickle

if __name__ == "__main__":
    pose = DWposeDetector()
    import cv2
    test_image = 'test_imgs/pose1.png'
    oriImg = cv2.imread(test_image)  # B,G,R order
    import matplotlib.pyplot as plt
    out, kp_vectors = pose(oriImg)
    plt.imsave('results/result.jpg', out)
    with open('results/result.pickle', 'wb') as handle:
        pickle.dump(kp_vectors, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
