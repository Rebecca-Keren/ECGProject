if __name__ == "__main__":
    glasses_on, glasses_off = get_img_lists('/home/dcor/datasets/CelebAMask-HQ/CelebAMask-HQ-attribute-anno.txt')

    all_images = np.concatenate((glasses_on, glasses_off))

    dataset_original = CustomDataSet(CELEB_A_DIR, TRANSFORM_IMG)

    agent = ModelAgentColorCorrection(dataset_original)

    agent.train()


