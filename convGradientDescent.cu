#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/time.h>
#include <assert.h>
#include <string.h>
#define memCheck(varname) { checkAllocationStatus((varname), #varname); } 

void checkAllocationStatus(void *ptr, const char *name) {
    if (ptr == NULL) {
        fprintf(stderr, "Error allocating memory for %s\n", name);
        exit(1);
    }
}


struct LabeledImage {
    float *data;
    int height;
    int width;
    float *label;
};

struct Dataset {
    struct LabeledImage **samples;
    int nImages;
};


/**
    * @brief Applies the sigmoid function to a value
    * 
    * @param x 
    * @return float 
    */
float sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

/**
    * @brief Applies the step function to a value
    * 
    * @param x 
    * @return float 
    */
float stepFunction(float x) {
    return (x > 0.5f) ? 1.0f : 0.0f;
}

/**
    * @brief Applies an activation function to a 1D array
    * 
    * @param features 
    * @param n 
    * @param activationFunction 
    */
void applyActivationFunction(float *features, int height, int width, float (*activationFunction)(float)) {

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            features[i * width + j] = (*activationFunction)(features[i * width + j]);
        }
    }

}

/**
    * @brief Computes the errors between predictions and labels
    * 
    * @param preds 
    * @param label 
    * @param height 
    * @param width 
    * @return float* 
    */
float* computeErrors(float *preds, float *label, int height, int width) {
    float *errors = (float *) malloc(height * width * sizeof(float)); memCheck(errors);

    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            errors[i * width + j] = preds[i * width + j] - label[i * width + j];
        }
    }

    return errors;
}

/**
    * @brief Initializes a kernel with random values
    * 
    * @param kernelHeight 
    * @param kernelWidth 
    * @return float* 
    */
float* initKernel(int kernelHeight, int kernelWidth) {
    float *kernel = (float *) malloc(kernelHeight * kernelWidth * sizeof(float)); memCheck(kernel);

    for (int i = 0; i < kernelHeight; i++) {
        for (int j = 0; j < kernelWidth; j++) {

            // generate random number from a normal distribution using Box-Muller transform
            /* float u = (float)rand() / RAND_MAX;
            float v = (float)rand() / RAND_MAX;
            float x = sqrt(-2.0 * log(u)) * cos(2.0 * M_PI * v); // Box-Muller transform
            kernel[i * kernelWidth + j] = x;
            */
            kernel[i * kernelWidth + j] = (float) (rand() % 100);
        }
    }
    return kernel;
}

/**
    * @brief Prints a 1D array
    * 
    * @param a 
    * @param n 
    */
void printArray(float *a, int n) {
    int i;
    for (i = 0; i < n; i++) printf("%8.2g ", a[i]);
    printf("\n");
}

/**
    * @brief Prints a 2D array
    * 
    * @param a 
    * @param m 
    * @param n 
    */
void printArray(float *a, int m, int n) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            printf("%8.2g ", a[i*n + j]);
        }
        printf("\n");
    }
}

/**
    * @brief Checks if two arrays are equal
    * 
    * @param a 
    * @param b 
    * @param n 
    */
void check(float *a, float *b, int m, int n) {
    int i, j;
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            if (a[i*n + j] != b[i*n + j]) {
                printf("Not Equal\n");
                return;
            }
        }
    }
    printf("Equal\n");
}

/**
    * @brief Creates a labeled image
    * 
    * @param height 
    * @param width 
    * @return struct LabeledImage* 
    */
struct LabeledImage *createLabeledImage(int height, int width) {
    // allocate image object
    struct LabeledImage *img = (struct LabeledImage*) malloc(sizeof(struct LabeledImage)); memCheck(img);
    
    // set attributes
    img->height = height;
    img->width = width;
    img->data = (float *) calloc(height * width, sizeof(float)); memCheck(img->data);
    img->label = (float *) calloc(height * width, sizeof(float)); memCheck(img->label);
    
    // generate random vertical edge
    // int randColumn = 1 + rand() % (width - 1) ;
    int randColumn = rand() % (width-1);
    for (int i = 0; i < height; i++) {
        for (int j = 0; j <= randColumn; j++) 
            img->data[i * width + j] = .25f;

        img->label[i * width + randColumn] = 1.0f;
     
        for (int j = randColumn + 1; j < width; j++)
            img->data[i * width + j] = 0.5f;
    }

    
    return img;
}

/**
    * @brief Creates a dataset of labeled images
    * 
    * @param nImages 
    * @param height 
    * @param width 
    * @return struct Dataset* 
    */
struct Dataset* createDataset(int nImages, int height, int width) {
    struct Dataset *dataset = (struct Dataset*) malloc(sizeof(struct Dataset)); memCheck(dataset);

    // set attributes
    dataset->nImages = nImages;
    dataset->samples = (struct LabeledImage**) malloc(nImages * sizeof(struct LabeledImage*)); memCheck(dataset->samples);

    // generate images   
    for (int idx = 0; idx < nImages; idx++) {
        dataset->samples[idx] = (struct LabeledImage*) malloc(sizeof(struct LabeledImage)); memCheck(dataset->samples[idx]);
        dataset->samples[idx] = createLabeledImage(height, width);
    }

    return dataset;
}

/**
    * @brief Prints a labeled image
    * 
    * @param img 
    */
void printLabeledImage(struct LabeledImage *img) {

    for (int i = 0; i < img->height; i++) {
        for (int j = 0; j < img->width; j++) {
            printf("%.3f  ", img->data[i * img->width + j]);
        }
        printf("\n");
    }
}

/**
    * @brief Prints a dataset
    * 
    * @param dataset 
    */
void printDataset(struct Dataset *dataset) {

    for (int idx = 0; idx < dataset->nImages; idx++) {
        printf("\n----\nSample n. %d\n", idx);
        printLabeledImage(dataset->samples[idx]);
        printf("----\n");
    }
}

/**
    * @brief Copies a dataset into a new dataset
    * 
    * @param ds 
    * @param nImages 
    * @param startIdx 
    * @return struct Dataset* 
    */
struct Dataset* copyDataset(struct Dataset *ds, int nImages, int startIdx) {
    struct Dataset *cp = (struct Dataset*) malloc(sizeof(struct Dataset)); memCheck(cp);

    // set attributes
    cp->nImages = nImages;
    cp->samples = (struct LabeledImage**) malloc(nImages * sizeof(struct LabeledImage*)); memCheck(cp->samples);
   
    for (int idx = startIdx, idx_cp = 0; idx < startIdx + nImages; idx++, idx_cp++) {

        // allocate img object
        cp->samples[idx_cp] = (struct LabeledImage*) malloc(sizeof(struct LabeledImage)); memCheck(cp->samples[idx_cp]);
        
        // get attributes
        int height = ds->samples[idx]->height;
        int width = ds->samples[idx]->width;
        float *data = ds->samples[idx]->data;
        float *label = ds->samples[idx]->label;

        // set attributes
        cp->samples[idx_cp]->height = height;
        cp->samples[idx_cp]->width = width;
        cp->samples[idx_cp]->data = (float *) malloc(height * width * sizeof(float)); memCheck(cp->samples[idx_cp]->data);
        cp->samples[idx_cp]->label = (float *) malloc(height * width * sizeof(float)); memCheck(cp->samples[idx_cp]->label);

        // copy data and label
        memcpy(cp->samples[idx_cp]->data, data, height * width * sizeof(float));
        memcpy(cp->samples[idx_cp]->label, label, height * width * sizeof(float));
    
    }

    return cp;
}

/**
    * @brief Splits a dataset into a training and a test set
    * 
    * @param dataset 
    * @param train 
    * @param test 
    * @param testSize 
    */
void trainTestSplitNaive(struct Dataset *dataset, struct Dataset **train, struct Dataset **test, float testSize) {

    int nTestImages = round((float) (testSize * dataset->nImages));
    int nTrainImages = dataset->nImages - nTestImages;

    printf("\n\nNumber of training images: %d (trainSize: %.2f)\nNumber of test images: %d (testSize: %.2f)\n\n", 
        dataset->nImages, nTrainImages, (float) 1.0f - testSize, nTestImages, testSize);

    // copy the training split into a new dataset
    *train = copyDataset(dataset, nTrainImages, 0);
    *test = copyDataset(dataset, nTestImages, nTrainImages);
}

/**
    * @brief Computes the dot product between two arrays
    * 
    * @param a 
    * @param b 
    * @param result 
    * @param n 
    */
void dotProduct(float *a, float *b, float *result, int n) {
    float value = 0.f;
    for (int i = 0; i < n; ++i) {
        value += (a[i] * b[i]);
    }
    *result = value;
}

/**
    * @brief Pads an image with zeros
    * 
    * @param input 
    * @param height 
    * @param width 
    * @param padh 
    * @param padw 
    * @param paddedHeight 
    * @param paddedWidth 
    * @return float* 
    */
float *pad(float *input, int height, int width, 
            int padh, int padw, 
            int *paddedHeight, int *paddedWidth) {
    
    // compute padded dimensions
    *paddedHeight = height + 2 * padh;
    *paddedWidth = width + 2 * padw;

    float *paddedInput = (float *) calloc((*paddedHeight) * (*paddedWidth), sizeof(float)); memCheck(paddedInput);

    for(int i = 0; i < height; i++) {
        for(int j = 0; j < width; j++) {
            paddedInput[(i + padh) * (*paddedWidth) + (j + padw)] = input[i * width + j];
        }
    }
    return paddedInput;
}

/**
    * @brief Performs the convolution operation between an image and a kernel
    * 
    * @param input 
    * @param inputHeight 
    * @param inputWidth 
    * @param padh 
    * @param padw 
    * @param kernel 
    * @param kernelHeight 
    * @param kernelWidth 
    * @param output 
    * @param outputHeight 
    * @param outputWidth 
    * @param bias 
    */
void convolve(float *input, int inputHeight, int inputWidth, int padh, int padw, 
                float *kernel, int kernelHeight, int kernelWidth,
                float *output,  int outputHeight, int outputWidth, 
                float bias) {
    // convolve(data, gradient, errors, k1: 6, k2: 6, height: 6, width: 6, padh: 1, padw: 1, 0)
    int paddedHeight, paddedWidth;
    float *paddedInput = pad(input, inputHeight, inputWidth, padh, padw, &paddedHeight, &paddedWidth);
    
    // printf("\npaddedHeight: %d, paddedWidth: %d\n", paddedHeight, paddedWidth);
    // printf("\nINPUT (after padding):\n");
    // printArray(paddedInput, paddedHeight, paddedWidth);
    
    // printf("\nCONVOLUTION:\n");
    // Perform the convolution operation
    for(int i = 0; i < outputHeight; i++) { // 6
        for(int j = 0; j < outputWidth; j++) { // 6
            float *tmp = (float *) malloc(kernelHeight * kernelWidth * sizeof(float)); memCheck(tmp);

            for(int m = 0; m < kernelHeight; m++) { // 6
                for(int n = 0; n < kernelWidth; n++) { // 6
                    tmp[m * kernelWidth + n] = paddedInput[(i + m) * paddedWidth + (j + n)];
                }
            }

            // printf("\nTMP:\n");
            // printArray(tmp, kernelHeight, kernelWidth);

            dotProduct(tmp, kernel, &output[i * outputWidth + j], kernelHeight * kernelWidth);
            output[i * outputWidth + j] += bias; // Add the bias after the dot product
            // printf("output[%d][%d] = %.2f\n\n", i, j, output[i * outputWidth + j]);

            free(tmp);
        }
    }
    // printf("\nEND CONVOLUTION:\n");

    free(paddedInput);
    
    // printf("OUT\n");
}

/**
    * @brief Computes the loss of the predictions
    * 
    * @param errors 
    * @param height 
    * @param width 
    * @return float 
    */
float getLoss(float *errors, int height, int width) {
    float loss = 0.f;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            // loss += errors[i * width + j]; // L1 loss
            loss += (powf(errors[i * width + j], 2)); // L2 loss
        }
    }
    
    return loss;
}

/**
    * @brief Computes the accuracy of the predictions
    * 
    * @param preds 
    * @param label 
    * @param height 
    * @param width 
    * @param accuracy 
    * @param nCorrect 
    */
void getAccuracy(float *preds, float *label, int height, int width, float *accuracy, int *nCorrect) {
    *accuracy = 0.f;
    *nCorrect = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (preds[i * width + j] == label[i * width + j]) (*nCorrect)++;
        }
    }

    *accuracy = (float) *nCorrect / (height * width);
    return;
}

/**
    * @brief Updates the kernel's weights
    * 
    * @param kernel 
    * @param gradient 
    * @param kernelHeight 
    * @param kernelWidth 
    * @param lr 
    */
void update(float *kernel, float* gradient, int kernelHeight, int kernelWidth, float lr) {
    for (int i = 0; i < kernelHeight; i++) {
        for (int j = 0; j < kernelWidth; j++) {
            kernel[i * kernelWidth + j] += (-lr * gradient[i * kernelWidth + j]);
        }
    }
}

/**
    * @brief Performs training of the convolutional layer given a training set and a kernel
    * 
    * @param trainingSet 
    * @param kernel 
    * @param kernelHeight 
    * @param kernelWidth 
    * @param epochs 
    * @param lr 
    * @param epochStep 
    */
void training(struct Dataset *trainingSet, float *kernel, int kernelHeight, int kernelWidth, int epochs, float lr, int epochStep) {
    // compute padding
    int padh = kernelHeight / 2;
    int padw = kernelWidth / 2;

    struct timeval convolveStart, convolveEnd;
    struct timeval applyActivationFunctionStart, applyActivationFunctionEnd;
    struct timeval computeErrorsStart, computeErrorsEnd;
    struct timeval getLossStart, getLossEnd;
    struct timeval gradientStart, gradientEnd;
    struct timeval updateStart, updateEnd;
    
    for (int epoch = 0; epoch < epochs; epoch++) {
        float trainLoss = 0.f;
        // printf("EPOCH: %d\n", epoch);
        for (int idx = 0; idx < trainingSet->nImages; idx++) {

            // get data
            float *data = trainingSet->samples[idx]->data;
            float *label = trainingSet->samples[idx]->label;
            int width = trainingSet->samples[idx]->width;
            int height = trainingSet->samples[idx]->height;
            
            //  printf("DATA: \n");
            // printArray(data, height, width);

            // printf("LABEL: \n");
            // printArray(label, height, width);
            
            // allocate feature maps (output of the conv layer)
            float *output = (float *) malloc(height * width * sizeof(float)); memCheck(output);
            
            // perform convolution between image (padded) and kernel
            gettimeofday(&convolveStart, NULL);
            convolve(data, height, width, padh, padw,
                        kernel, kernelHeight, kernelWidth,
                        output, height, width, 0);
            gettimeofday(&convolveEnd, NULL);
            
            // printf("\nOUTPUT (after conv):\n");
            // printArray(output, height, width);
            
            // activations
            gettimeofday(&applyActivationFunctionStart, NULL);
            applyActivationFunction(output, height, width, sigmoid);
            gettimeofday(&applyActivationFunctionEnd, NULL);

            // printf("\nOUTPUT (after activations):\n");
            // printArray(output, height, width);
            
            // compute error
            gettimeofday(&computeErrorsStart, NULL);
            float *errors = computeErrors(output, label, height, width); memCheck(errors);
            gettimeofday(&computeErrorsEnd, NULL);

            // printf("\nERRORS: \n");
            // printArray(errors, height, width);


            // compute loss
            gettimeofday(&getLossStart, NULL);
            float loss = getLoss(errors, height, width);
            gettimeofday(&getLossEnd, NULL);
            

            // printf("loss = %.2f\n", loss);
            trainLoss += loss;

            // compute gradient
            float *gradient = (float *) malloc(kernelHeight * kernelWidth * sizeof(float)); memCheck(gradient);
            
            gettimeofday(&gradientStart, NULL);
            convolve(data, height, width, padh, padw, 
                        errors, height, width,
                        gradient, kernelHeight, kernelWidth, 0);
            gettimeofday(&gradientEnd, NULL);

            // printf("\nGRADIENT (after conv):\n");
            // printArray(gradient, k1, k2);

            // printf("\nKERNEL (before update):\n");
            // printArray(kernel, k1, k2);
            
            
            // update kernel's weights
            gettimeofday(&updateStart, NULL);
            update(kernel, gradient, kernelHeight, kernelWidth, lr);
            gettimeofday(&updateEnd, NULL);
            
            // printf("\nKERNEL (after update):\n");
            // printArray(kernel, k1, k2);
            
            // deallocate dynamic variables
            free(output);                   
            free(errors);
            free(gradient);
        }

        // average loss per epoch
        trainLoss /= trainingSet->nImages;

        // print loss
        if (epoch == 0 || (epoch+1) % epochStep == 0) {
            printf("[INFO] epoch=%d, loss=%.4f ", epoch+1, trainLoss);
            printf("(convolve: %.2es, applyActivationFunction: %.2es, computeErrors: %.2es, getLoss: %.2es, gradient: %.2es, update: %.2es)\n", 
                (double) ((convolveEnd.tv_sec - convolveStart.tv_sec) + (convolveEnd.tv_usec - convolveStart.tv_usec) / 1000000.0f),
                (double) ((applyActivationFunctionEnd.tv_sec - applyActivationFunctionStart.tv_sec) + (applyActivationFunctionEnd.tv_usec - applyActivationFunctionStart.tv_usec) / 1000000.0f),
                (double) ((computeErrorsEnd.tv_sec - computeErrorsStart.tv_sec) + (computeErrorsEnd.tv_usec - computeErrorsStart.tv_usec) / 1000000.0f),
                (double) ((getLossEnd.tv_sec - getLossStart.tv_sec) + (getLossEnd.tv_usec - getLossStart.tv_usec) / 1000000.0f),
                (double) ((gradientEnd.tv_sec - gradientStart.tv_sec) + (gradientEnd.tv_usec - gradientStart.tv_usec) / 1000000.0f),
                (double) ((updateEnd.tv_sec - updateStart.tv_sec) + (updateEnd.tv_usec - updateStart.tv_usec) / 1000000.0f)
            );
        }
    }
}

/**
    * @brief Predicts the output of the convolutional layer given a test set and a kernel
    * 
    * @param testSet 
    * @param kernel 
    * @param kernelHeight 
    * @param kernelWidth 
    */
void predict(struct Dataset *testSet, float *kernel, int kernelHeight, int kernelWidth, int interactive) {

    struct timeval convolveStart, convolveEnd;
    struct timeval applyActivationFunctionStart, applyActivationFunctionEnd;
    struct timeval getAccuracyStart, getAccuracyEnd;

    int correctPreds = 0;
    int totalPreds = 0;
    float accuracy;
    int nCorrect;
    float testAccuracy = 0.0f;

    // compute padding
    int padh = kernelHeight / 2;
    int padw = kernelWidth / 2;

    
    for (int idx = 0; idx < testSet->nImages; idx++) {

        if (interactive) printf("\n----\nTest Sample n. %d\n", idx);
        // get data
        float *data = testSet->samples[idx]->data;
        float *label = testSet->samples[idx]->label;
        int width = testSet->samples[idx]->width;
        int height = testSet->samples[idx]->height;
        
        if (interactive) { printf("DATA: \n"); printArray(data, height, width); }
        if (interactive) { printf("LABEL: \n"); printArray(label, height, width); }
        
        // allocate feature maps (output of the conv layer)
        float *preds = (float *) malloc(height * width * sizeof(float)); memCheck(preds);

        // perform convolution between image (padded) and kernel
        gettimeofday(&convolveStart, NULL);
        convolve(data, height, width, padh, padw, 
                    kernel, kernelHeight, kernelWidth,
                    preds, height, width, 0);
        gettimeofday(&convolveEnd, NULL);
        
        // activations
        gettimeofday(&applyActivationFunctionStart, NULL);
        applyActivationFunction(preds, height, width, stepFunction);
        gettimeofday(&applyActivationFunctionEnd, NULL);

        if (interactive) { printf("\nPREDS: \n"); printArray(preds, height, width); }
        
        // get accuracy
        gettimeofday(&getAccuracyStart, NULL); 
        getAccuracy(preds, label, height, width, &accuracy, &nCorrect);
        gettimeofday(&getAccuracyEnd, NULL);

        testAccuracy += accuracy;
        correctPreds += nCorrect;
        totalPreds += height * width;
        printf("[INFO] (%d/%d) accuracy=%.2f (%d/%d) ", idx+1, testSet->nImages, accuracy * 100, nCorrect, height * width);
        printf("(convolve: %.2es, applyActivationFunction: %.2es, getAccuracy: %.2es)\n", 
            (double) ((convolveEnd.tv_sec - convolveStart.tv_sec) + (convolveEnd.tv_usec - convolveStart.tv_usec) / 1000000.0f),
            (double) ((applyActivationFunctionEnd.tv_sec - applyActivationFunctionStart.tv_sec) + (applyActivationFunctionEnd.tv_usec - applyActivationFunctionStart.tv_usec) / 1000000.0f),
            (double) ((getAccuracyEnd.tv_sec - getAccuracyStart.tv_sec) + (getAccuracyEnd.tv_usec - getAccuracyStart.tv_usec) / 1000000.0f)
        );
        // deallocate dynamic variables
        free(preds);

        if (interactive) { fflush(stdin); printf("\nPress ENTER to continue...\n"); getc(stdin); }
    }

    // average accuracy
    testAccuracy /= testSet->nImages;
    
    printf("[INFO] accuracy=%.2f (%d/%d) ", testAccuracy * 100, correctPreds, totalPreds);
    

}


int main(int argc, char *argv[]) {
    srand(time(NULL));
    struct timeval trainingStart, trainingEnd;
    struct timeval testStart, testEnd;

    // hyperparameters
    int nImages     = 500;
    int height      = 12;
    int width       = 12;
    float testSize = 0.1f;

    // training hyperparameters
    int epochs = 100;
    int epochStep = 10;
    float lr = 0.1f;
    // int bs = 32;
    int kernelSize = 3;

    int interactive = 0;
    int opt;
    while ((opt = getopt(argc, argv, "n:h:w:k:e:s:l:t:i:h")) != -1) {
        switch (opt) {
        case 'n':
            nImages = atoi(optarg);
            if (nImages <= 0) {
                fprintf(stderr, "Number of images must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'h':
            height = atoi(optarg);
            if (height <= 0) {
                fprintf(stderr, "Image height must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'w':
            width = atoi(optarg);
            if (width <= 0) {
                fprintf(stderr, "Image width must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'k':
            kernelSize = atoi(optarg);
            if (kernelSize <= 0) {
                fprintf(stderr, "Kernel size must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            if (kernelSize > height || kernelSize > width) {
                fprintf(stderr, "Kernel size must be smaller than image size\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'e':
            epochs = atoi(optarg);
            if (epochs <= 0) {
                fprintf(stderr, "Number of epochs must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 's':
            epochStep = atoi(optarg);
            if (epochStep <= 0) {
                fprintf(stderr, "Epoch step must be greater than 0\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'l':
            lr = atof(optarg);
            break;
        case 't':
            testSize = atof(optarg);
            if (testSize <= 0.f || testSize >= 1.f) {
                fprintf(stderr, "Test size must be between 0 and 1\n");
                exit(EXIT_FAILURE);
            }
            break;
        case 'i':
            interactive = atoi(optarg);
            if (interactive != 0 && interactive != 1) {
                fprintf(stderr, "Interactive mode must be either 0 or 1\n");
                exit(EXIT_FAILURE);
            }
            break;
        case '?':
            fprintf(stderr, "Usage: %s [-n nImages] [-h height] [-w width] [-k kernelSize] [-e epochs] [-s epochStep] [-l lr] [-t testSize] [-i interactive]\n", argv[0]);
            exit(EXIT_FAILURE);
        case 'H':
            printf("Usage: %s [-n nImages] [-h height] [-w width] [-k kernelSize] [-e epochs] [-s epochStep] [-l lr] [-t testSize] [-i interactive]\n", argv[0]);
            exit(EXIT_SUCCESS);
        }
    }

    printf("\nHyperparameters:\n");
    printf("Number of Images: %d\n", nImages);
    printf("Image Height: %d\n", height);
    printf("Image Width: %d\n", width);
    printf("Kernel Size: %d\n", kernelSize);
    printf("Number of Epochs: %d\n", epochs);
    printf("Epoch Step: %d\n", epochStep);
    printf("Learning Rate: %.3f\n", lr);

    // create dataset
    struct Dataset *dataset = createDataset(nImages, height, width);

    // print dataset
    // printf("DATASET: \n");
    // printDataset(dataset);

    // define training and test split
    struct Dataset *train;
    struct Dataset *test;
    trainTestSplitNaive(dataset, &train, &test, testSize);

    //printf("TRAINING SPLIT: \n");
    //printDataset(train);

    //printf("TEST SPLIT: \n");
    //printDataset(test);

    // initialize kernel
    float *kernel = initKernel(kernelSize, kernelSize);
    // float kernel[] = {0.2782423, 1.14216316, 0.47450542, -0.02898605, 0.54504466, 0.24390744, -1.46459284, 0.03084573, -1.18414616};
    printf("\n----\nKERNEL (before training): \n");
    printArray(kernel, kernelSize, kernelSize);
    printf("----\n");

    
    // perform training
    printf("----\nTRAINING: \n");
    gettimeofday(&trainingStart, NULL);
    training(dataset, kernel, kernelSize, kernelSize, epochs, lr, epochStep);
    gettimeofday(&trainingEnd, NULL);
    printf("Total elapsed time for training: %.4f seconds\n", (double) ((trainingEnd.tv_sec - trainingStart.tv_sec) + (trainingEnd.tv_usec - trainingStart.tv_usec) / 1000000.0f));

    // print learned kernel
    printf("\n----\nKERNEL (after training): \n");
    printArray(kernel, kernelSize, kernelSize);
    printf("----\n");

    fflush(stdin);
    printf("Press ENTER to continue...\n");
    getc(stdin);

    // perform training
    printf("----\nTEST: \n");
    gettimeofday(&testStart, NULL);
    predict(test, kernel, kernelSize, kernelSize, interactive);
    gettimeofday(&testEnd, NULL);
    printf("\n----\n");
    printf("Total elapsed time for test: %.4f seconds\n", (double) ((testEnd.tv_sec - testStart.tv_sec) + (testEnd.tv_usec - testStart.tv_usec) / 1000000.0f));

    exit(EXIT_SUCCESS);
}