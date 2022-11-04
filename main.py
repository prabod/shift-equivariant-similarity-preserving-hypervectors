import logging
import time

import numpy as np
from scipy.sparse import csr_matrix
from skimage.util import view_as_windows
from tqdm import tqdm

from sesph import make_codevector_matrix, get_lbp_features, make_sparsevector

INV_flag = 0

CDT_type0 = 0
CDT_K0 = 60
CDT_type = 0
CDT_K = 60

MakePrototypes = 1
splitTest2parts = 1

flag_binary0 = 0
LimNum1s = 50000
LimNum1s2nd = 2000
NumTrainSamples = 60000
NumTestSamples = 10000
MultiClassTraining = 1
NumberOfClasses = 10
TDS_start = 0.3
TDS_step = 0.05
TDS_end = 0.3

shy = 0
shy_step = 6
shx = 0
shx_step = 6

shiftCrossFlag = 0

deskew = 0
NumOfTrainingCycles = 200
StopTrainErrors = 1

Dc_start = 4
Dc_step = 1
Dc_end = 4
Dc_ratioXY_start = 1
Dc_ratioXY_step = 1
Dc_ratioXY_end = 1

Dc2_start = 1
Dc2_step = 1
Dc2_end = 1

seed = 101
seed_start = seed
number_experiments = 1
seed_finish = seed + number_experiments - 1

y_dim, x_dim = 38, 38
y2_dim, x2_dim = 511, 1
N = 500000
cyc = 1
p = 1


def get_lbp_dataset(images, mask_idx):
    train_expanded = np.pad(images, ((0, 0), (2, 2), (2, 2), (0, 0)), 'constant', constant_values=np.nan)
    train_patches = view_as_windows(train_expanded, (1, 5, 5, 1))[..., 0, :, :, 0]
    train_patches = np.squeeze(train_patches, axis=3)
    train_lbp = get_lbp_features(train_patches, mask_idx)
    train_lbp = np.pad(train_lbp, ((0, 0), (5, 5), (5, 5)), 'constant', constant_values=0)
    return train_lbp


def get_sparse_vector_dataset(dataset_lbp, num_samples, num1_limit):
    train_s_rows = []
    train_s_cols = []
    for m, a in enumerate(tqdm(dataset_lbp)):
        a_nonzero = np.nonzero(a)
        arr1 = np.array([a_nonzero[0], a_nonzero[1]])
        a_values = a[a_nonzero]
        arr2 = np.array([[0] * a_nonzero[1].shape[0], a_values])
        sparse_vec = make_sparsevector(arr1, arr2, c_m, num1_limit, m)
        train_s_rows += sparse_vec[0]
        train_s_cols += sparse_vec[1]

    train_sparse = csr_matrix(([1] * len(train_s_cols), (train_s_rows, train_s_cols)),
                              shape=(num_samples, N),
                              dtype=np.bool8
                              )
    return train_sparse


if __name__ == "__main__":
    logging.basicConfig(filename='%d.log' % time.time(), level=logging.INFO)
    train_images = np.load("mnist_c_identity/train_images.npy")
    train_labels = np.load("mnist_c_identity/train_labels.npy")

    test_images = np.load("mnist_c_identity/test_images.npy")
    test_labels = np.load("mnist_c_identity/test_labels.npy")

    row = []
    col = []
    data = np.array([1] * 9)
    for i in range(0, 5, 2):
        for j in range(0, 5, 2):
            row.append(i)
            col.append(j)
    row = np.array(row)
    col = np.array(col)
    mask = csr_matrix((data, (row, col)), shape=(5, 5)).toarray()

    c_m = make_codevector_matrix(y_dim, x_dim, Dc_start, Dc_ratioXY_start, N, y2_dim, x2_dim, Dc2_start, Dc2_start, p,
                                 cyc, seed)

    t1 = time.time()
    train_lbp = get_lbp_dataset(train_images, (row, col))
    t2 = time.time()
    logging.info("Extracting training LBP features took %f seconds" % (t2 - t1))
    t1 = time.time()
    test_lbp = get_lbp_dataset(test_images, (row, col))
    t2 = time.time()
    logging.info("Extracting testing LBP features took %f seconds" % (t2 - t1))

    t1 = time.time()
    train_sparse = get_sparse_vector_dataset(train_lbp, NumTrainSamples, LimNum1s2nd)
    t2 = time.time()
    logging.info("Creating sparse vectors for training dataset took %f seconds" % (t2 - t1))
    t1 = time.time()
    test_sparse = get_sparse_vector_dataset(test_lbp, NumTestSamples, LimNum1s2nd)
    t2 = time.time()
    logging.info("Creating sparse vectors for testing dataset took %f seconds" % (t2 - t1))

    logging.info("Training ...")
    for Dc2 in range(Dc2_start, Dc2_end + 1, Dc2_step):
        Dc2_ratioXY = 1 / Dc2
        for Dc in range(Dc_start, Dc_end + 1, Dc_step):
            for Dc_ratioXY in range(Dc_ratioXY_start, Dc_ratioXY_end + 1, Dc_ratioXY_step):
                TestErrorsIterIseed = []
                for seed in range(seed_start, seed_finish + 1):
                    iseed = seed - seed_start + 1

                    LimitMatrix = 0  # 1 use [0, ~360000] for weights, 0 - use + and - weights
                    TrainSparse = -1
                    code_sparse_flag = 1
                    p_drop = 0.0  # dropout
                    cast_do = 1
                    maxnorm_flag = -1  # 2 #-1

                    UseReadyMatrix = 0  # 1 - Use already trained matrix from file
                    ContinueTrainMatrix = 0  # 1 - Continue training using matrix from file. Requires UseReadyMatrix = 0
                    StopTrain = 0.0  # fraction of first iteration errors to stop training # 0.01

                    # Training
                    # TDS = 0.7 # initial "training defense space"
                    TDS_decay = 0.009  # decreasing TDS #0.95 #0.9 #new version 18.01.2010
                    TDS_iter = 200  # 50 # 5#??decreasing each TDS_iter iterations of training
                    delta = 1.0  # matrix increment - decrement step
                    # NumOfTrainingCycles = 50 #50
                    # Changed number of samples per iteration from mean to max to 14.01
                    InitMatrix = 0  # Initial matrix values. 0 or 1

                    cycleBegin = 1
                    cycleEnd = 1  # è

                    for cycle in range(cycleBegin, cycleEnd + 1):
                        TestErrorsIterSplit = []
                        NumTestSamplesPart = NumTestSamples / splitTest2parts
                        num_shx = np.arange(-shx, shx + 1, shx_step)
                        num_shy = np.arange(-shy, shy + 1, shy_step)
                        num_shifts = num_shx.shape[-1] * num_shy.shape[-1]

                        for spl in range(splitTest2parts + 1):
                            spVectors0TestShifts = []
                            spVectors0Train_Lim = []
                            for TDS in np.arange(TDS_start, TDS_end + TDS_step, TDS_step):
                                for Winner4shifts in range(1):
                                    for thrWinners in range(-1000, -999):
                                        TheActivations0 = np.zeros([NumberOfClasses, 1])
                                        TheActivations = np.zeros([NumberOfClasses, 1])
                                        TheClasses = np.arange(NumberOfClasses)

                                        currentTDS = TDS
                                        TrainingErrors = 0
                                        TrainingCorrect = 0
                                        tdsind = 0
                                        TheMatrix = np.zeros((NumberOfClasses, N), dtype=np.float32)
                                        testErrorIter = []
                                        perm = np.arange(NumTrainSamples)
                                        for tc in tqdm(range(NumOfTrainingCycles)):
                                            TrainingErrors = 0
                                            TrainingCorrect = 0
                                            if MakePrototypes == 1 and tc == 0:
                                                for ppp in range(NumberOfClasses):
                                                    TheMatrix[ppp, :] = train_sparse[train_labels == ppp].sum(
                                                        axis=0) / NumberOfClasses
                                            else:
                                                tce = 0
                                                ClassLabels = np.copy(train_labels)
                                                for fc in range(NumTrainSamples):
                                                    sample_num = perm[fc]
                                                    if code_sparse_flag == 1:
                                                        x = train_sparse[sample_num].A
                                                        TheActivations0 = (TheMatrix @ x.T).flatten()
                                                    if maxnorm_flag == -1:
                                                        TheActivations = TheActivations0

                                                    TheTrueClass = ClassLabels[fc]
                                                    TheActivations[TheTrueClass] = TheActivations[TheTrueClass] * (
                                                            1.0 - currentTDS)
                                                    ind2unlearn = TheActivations > TheActivations[TheTrueClass]

                                                    if TheActivations[0] > TheActivations[1]:
                                                        TheFirstActivation = TheActivations[0]
                                                        TheFirstClass = TheClasses[0]
                                                    else:
                                                        TheFirstActivation = TheActivations[1]
                                                        TheFirstClass = TheClasses[1]

                                                    for i in range(1, NumberOfClasses):
                                                        if TheActivations[i] > TheFirstActivation:
                                                            TheFirstActivation = TheActivations[i]
                                                            TheFirstClass = TheClasses[i]
                                                    if code_sparse_flag == 1:
                                                        if TheFirstClass != TheTrueClass:
                                                            TrainingErrors += 1
                                                            tce += 1
                                                            if MultiClassTraining == 1:
                                                                TheMatrix[TheTrueClass, x.flatten()] += delta
                                                                num_wrong_class = sum(ind2unlearn.flatten())
                                                                if num_wrong_class > 0:
                                                                    for r in ind2unlearn.flatten().nonzero()[0]:
                                                                        TheMatrix[
                                                                            r, x.flatten()] -= delta / num_wrong_class
                                                                else:
                                                                    TheMatrix[:, x.flatten()] -= delta / 10
                                                        else:
                                                            TrainingCorrect += 1

                                            logging.info("Epoch : %d, Train Correct: %d, Train Error: %d" % (
                                                tc, TrainingCorrect, TrainingErrors))

                                            TestErrors = 0
                                            TestCorrect = 0
                                            TestErrorsRel = 0
                                            TestCorrectRel = 0
                                            TheActivations = np.zeros([NumberOfClasses, 1])
                                            TheClasses = np.arange(NumberOfClasses)
                                            ClassLabelsTest = test_labels

                                            t1 = time.time()
                                            for ii in range(NumTestSamples):
                                                x = test_sparse[ii].A
                                                TheActivations = (TheMatrix @ x.T).flatten()
                                                TheTrueClass = ClassLabelsTest[ii]
                                                TheFirstActivation = TheActivations[0]
                                                TheFirstClass = TheClasses[0]
                                                for i in range(1, NumberOfClasses):
                                                    if TheActivations[i] > TheFirstActivation:
                                                        TheFirstActivation = TheActivations[i]
                                                        TheFirstClass = TheClasses[i]
                                                if TheFirstClass != TheTrueClass:
                                                    TestErrors = TestErrors + 1
                                                else:
                                                    TestCorrect = TestCorrect + 1
                                            t2 = time.time()
                                            logging.info("Epoch : %d, Test Correct: %d, Test Error: %d" % (
                                                tc, TestCorrect, TestErrors))
                                            tdsind = tdsind + 1
                                            if tdsind >= TDS_iter:
                                                currentTDS = currentTDS * (1.0 - TDS_decay)
                                                tdsind = 0
