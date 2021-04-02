#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <string.h>

#include <inttypes.h>
#include <stdint.h>
#include <stddef.h>

#define false 0
#define true 1

#define bool uint_fast8_t
#define b8   uint8_t
#define b16  uint16_t
#define b32  uint32_t
#define b64  uint64_t

#define s8  int8_t
#define s16 int16_t
#define s32 int32_t
#define s64 int64_t

#define u8  uint8_t
#define u16 uint16_t
#define u32 uint32_t
#define u64 uint64_t

#undef max
#undef min

#ifdef DEBUG
#define alloc(size) calloc(1, (size))
#else
#define alloc(size) malloc((size))
#endif

// I don't like this, but since variable length arrays aren't portable, it's probably the best available solution
// an alternative would be to malloc anything that uses this, but allocating and freeing like that is really slow
// a better alternative would be to provide a memory arena that preallocates large chunks of memory. in all cases where this constant is used, a stack based allocator would work fine, but I didn't feel like doing that
#define MAX_CLASSES 512

//~ NOTE(mdizdar): forward declarations
struct Node;
struct Data;

double evaluate_fmeasure(struct Node *, struct Data *);
//~

//~ NOTE(mdizdar): these are controllable by passing commandline arguments, these are the default values
u64 INSTANCES = 1024ULL; // population size per generation
u64 N_CROSSOVER = 512; // number of offspring to create by crossover
u64 N_MUTATION = 512; // number of mutated clones to create

u64 NUM_THREADS = 1; // number of threads used for evaluating all instances
u64 MAX_DEPTH = 10; // max depth of each tree

u64 MAX_MUTATIONS = 10; // max number of mutations to perform on each mutated clone (keep in mind trees have up to 2^MAX_DEPTH nodes)
u64 MAX_GENERATIONS = 3000; // number of generations to simulate
u64 TIMES = 1; // how many times to run the whole simulation (reshuffling the whole dataset, but not reloading it). does not reload the loaded trees from a checkpoint file.

u64 CHECKPOINTS = 0; // if this value is 0 it doesn't create any checkpoints and prints stats for all generations. other values mean save checkpoint and print stats every CHECKPOINTS generations.

u32 SEED = 420;

const char *FILENAME = NULL; // the path the dataset file, must be in the correct format (i.e. the output of that stupid python script; because I couldn't be bothered to write PCA in C)
const char *CONTINUE_FILE = NULL; // the path to a checkpoint file from which to continue training. continuing on a differently shaped dataset (number of classes, number of features) is undefined behavior.
const char *EVAL_ONLY_PATH = NULL; // path to checkpoint file which we wish to only evaluate without continuing training
const char *OUTPUT_PATH = NULL; // if an output file is specified, the final evaluation will be printed there instead of to stdout
const char *CHECKPOINT_PATH = ""; // path to a directory where checkpoints will be stored as checkpoint_####.trees

bool STRATIFY = false; // whether to stratify the shuffled dataset so there's a roughly equal ratio of each class in the training and testing datasets
bool PRUNE = false; // whether to prune the trees after crossover and mutation, shouldn't affect fitness, trees should become smaller, program execution might suffer greatly
bool SILENT = false; // when true, doesn't print anything (except errors and warnings) to stdout.

double SPLIT_RATIO = 0.8; // the percentage of the dataset alloted to training, the rest is for testing.
double TURN_TO_LEAF = 0.33; // the percentage with which a randomly selected node will mutate into a leaf node.

double (*FITNESS)(struct Node *, struct Data *) = evaluate_fmeasure; // function pointer to a fitness function
u8 SORTING = 0; // whether the members of a population should be sorted with fitness ascending or descending - should be set while selecting the function itself
//~

//~ NOTE(mdizdar): some constants
const u64 MIN_DEPTH = 2; // the minimum depth of a tree, to prevent it from being a stub or NULL
const u64 DEFAULT_CLASS = 0; // actually useless, doesn't do anything, but we tolerate it anyway
const double EPSILON = 1e-7; // when comparing floats, if the absolute value of the difference between them is less than EPSILON we'll say they're equal. also used to trick the log function into taking the log of 0.
const double LOG2_EPSILON = -26.5754247591; // log2(1e-7); - used when calculating crossentropy because actually taking the log is slower
//~

//~ NOTE(mdizdar): this should be a struct I think
// just data about the dataset that I like being global
double *feature_min;
double *feature_max;
bool *feature_discrete;
u64 CLASSES, FEATURES;
//~

struct Node **SORT_HELPER; // the lone uncategorized global. used in merge sort. global because mallocing it every time we want to sort is just not gonna happen

//~ NOTE(mdizdar): structures
// just for legibility I guess
typedef enum Sorting {
    Descending = 0,
    Ascending = 1
} Sorting;

// contains either the "index" of a discrete value's attribute, or a floating point continuous value (continuous integers are treated the same)
typedef union {
    u64 attribute_index;
    double value;
} Feature;

// a single row of data, contains an array of features and a label
typedef struct {
    Feature *data;
    u64 label;
} Row;

// a dataset, contains an array of rows and the total size of it
typedef struct Data {
    Row *data;
    u64 size;
} Data;

// decision tree node, contains information of its own fitness, either the feature it splits on and its value, or the class it predicts if it's a leaf node. has pointers to two children and its parent. the root's parent is NULL, the children of leaf nodes are NULL
typedef struct Node {
    union {
        struct {
            u64 feature;
            //bool is_binary_split;
            Feature split;
        };
        u64 discrete_value;
    };
    double fitness;
    struct Node *parent;
    struct Node *left; // TODO(mdizdar): maybe we'll want the option of not having just a binary tree
    struct Node *right;
} Node;
//~

//~ NOTE(mdizdar): "generic" helper functions

static inline u64 max(u64 a, u64 b) {
    return a > b ? a : b;
}

static inline u64 min(u64 a, u64 b) {
    return a < b ? a : b;
}

// Fisher-Yates shuffle I guess
void shuffle(void *array, size_t n, size_t size) {
    char tmp[sizeof(Row)];
    char *arr = array;
    size_t stride = size * sizeof(char);
    
    if (n > 1) {
        for (size_t i = 0; i < n - 1; ++i) {
            size_t rnd = (size_t) rand();
            size_t j = i + rnd / (RAND_MAX / (n - i) + 1);
            
            memcpy(tmp, arr + j * stride, size);
            memcpy(arr + j * stride, arr + i * stride, size);
            memcpy(arr + i * stride, tmp, size);
        }
    }
}

// these disgusting defines are for readability, thank me later
#define TRAIN_COUNT(x) train_count[dataset->data[(x)].label]
#define TEST_COUNT(x) test_count[dataset->data[(x)].label]
#define GOOD_COUNT(x) good_count[dataset->data[(x)].label]

// (hopefully) stratifies the dataset so that ratio% of each class ends up in the training set
void stratify(Data *dataset, double ratio) {
    const u64 split = (u64)(dataset->size * ratio);
    u64 train_count[MAX_CLASSES] = {0};
    u64 test_count[MAX_CLASSES] = {0};
    for (u64 i = 0; i < split; ++i) {
        ++TRAIN_COUNT(i);
    }
    for (u64 i = split; i < dataset->size; ++i) {
        ++TEST_COUNT(i);
    }
    double good_count[MAX_CLASSES] = {0};
    for (u64 i = 0; i < CLASSES; ++i) {
        good_count[i] = (train_count[i] + test_count[i]) * ratio;
    }
    
    char tmp[sizeof(Row)];
    for (u64 i = 0, j = split, k = split; i < split; ++i) {
        if (TRAIN_COUNT(i) >= GOOD_COUNT(i) && (TRAIN_COUNT(i)-1) < GOOD_COUNT(i)) {
            // we're good here
            continue;
        }
        if (TRAIN_COUNT(i) > GOOD_COUNT(i)) {
            for (; j < dataset->size; ++j) {
                if (TRAIN_COUNT(j) > GOOD_COUNT(j)) {
                    // we don't need this one
                    continue;
                }
                // swap
                --TRAIN_COUNT(i); ++TRAIN_COUNT(j);
                ++TEST_COUNT(i); --TEST_COUNT(j);
                memcpy(tmp, &dataset->data[j], sizeof(Row));
                memcpy(&dataset->data[j], &dataset->data[i], sizeof(Row));
                memcpy(&dataset->data[i], tmp, sizeof(Row));
                ++j;
                break;
            }
        } else { // if (TRAIN_COUNT(i) < GOOD_COUNT(i)
            for (; k < dataset->size; ++k) {
                if (TRAIN_COUNT(k) < GOOD_COUNT(k)) {
                    // we don't need this one
                    continue;
                }
                // swap
                --TRAIN_COUNT(i); ++TRAIN_COUNT(k);
                ++TEST_COUNT(i); --TEST_COUNT(k);
                memcpy(tmp, &dataset->data[k], sizeof(Row));
                memcpy(&dataset->data[k], &dataset->data[i], sizeof(Row));
                memcpy(&dataset->data[i], tmp, sizeof(Row));
                ++k;
                break;
            }
        }
    }
}
#undef TRAIN_COUNT
#undef TEST_COUNT
#undef GOOD_COUNT

// merge sort
// NOTE(mdizdar): depends on the SORT_HELPER array being large enough (and allocated)
void sort(Node *members[], u64 n, Sorting sorting) {
    for (u64 width = 1; width < n; width <<= 1) {
        for (u64 i = 0; i < n; i = i + 2*width) {
            u64 iLeft = i, ii = iLeft, iRight = min(i+width, n), jj = iRight, iEnd = min(i+2*width, n);
            for (u64 k = iLeft; k < iEnd; ++k) {
                if (ii < iRight && (jj >= iEnd || (sorting == Ascending && members[ii]->fitness <= members[jj]->fitness) || (sorting == Descending && members[ii]->fitness >= members[jj]->fitness))) {
                    SORT_HELPER[k] = members[ii];
                    ++ii;
                } else {
                    SORT_HELPER[k] = members[jj];
                    ++jj;
                }
            }
        }
        for (u64 i = 0; i < n; ++i) {
            members[i] = SORT_HELPER[i];
        }
    }
}

// just counts the number of digits in an u64
static inline u64 digits(u64 x) {
    if (!x) return 1;
    int ret = 0;
    while (x) {
        ++ret;
        x /= 10;
    }
    return ret;
}
//~

//~ NOTE(mdizdar): basic operations on trees

// creates a randomised leaf node; doesn't set its children to NULL (DO NOT FORGET TO DO THAT)
static inline void random_leaf(Node *node) {
    node->discrete_value = rand() % CLASSES;
}

// creates a randomised internal node; does nothing with the children - be careful about that
static inline void random_node(Node *node) {
    node->feature = rand() % FEATURES;
    if (feature_discrete[node->feature]) {
        node->split.attribute_index = rand() % (u64)feature_max[node->feature];
    } else {
        double range = feature_max[node->feature] - feature_min[node->feature];
        node->split.value = range*rand()/RAND_MAX + feature_min[node->feature];
    }
}

// creates a randomised tree of a set depth by recursively creating each child node. this will create a complete binary tree, maybe that's not a good thing tho :shrug:
// if called with a root node, it'll expect parent to be NULL, but if not it'll work anyway
Node *create_random_tree(Node *root, Node *parent, u64 depth) {
    root->parent = parent;
    if (depth == 0) {
        // is leaf
        random_leaf(root);
        root->left = NULL;
        root->right = NULL;
    } else {
        random_node(root);
        root->left = create_random_tree(alloc(sizeof(Node)), root, depth-1);
        root->right = create_random_tree(alloc(sizeof(Node)), root, depth-1);
    }
    return root;
}

// copies all of the information stored in a tree and creates a new tree
Node *copy_tree(Node *root, Node *parent) {
    Node *new_root = alloc(sizeof(Node));
    new_root->parent = parent;
    if (root->left == NULL && root->right == NULL) {
        new_root->discrete_value = root->discrete_value;
        new_root->left = NULL;
        new_root->right = NULL;
    } else {
        new_root->feature = root->feature;
        // NOTE(mdizdar) this is a redundant if statement since attribute_index and value both take up 8 bytes of memory and copying one means copying the other - but whatever
        if (feature_discrete[root->feature]) {
            new_root->split.attribute_index = root->split.attribute_index;
        } else {
            new_root->split.value = root->split.value;
        }
        new_root->left = copy_tree(root->left, new_root);
        new_root->right = copy_tree(root->right, new_root);
    }
    return new_root;
}

// find a random node that isn't the root. usually called with 1/count(root) for the choice_rate
Node *find_random_node(Node *root, bool is_root, double choice_rate) {
    if (!is_root && 1.*rand()/RAND_MAX < choice_rate) {
        return root;
    }
    if (root->left == NULL && root->right == NULL) {
        return root;
    }
    if (rand() < RAND_MAX/2) {
        return find_random_node(root->left, false, choice_rate);
    } else {
        return find_random_node(root->right, false, choice_rate);
    }
}

u64 depth(Node *root) {
    if (!root) return 0;
    return 1+max(depth(root->left), depth(root->right));
}

u64 count(Node *root) {
    if (!root) return 0;
    return 1+count(root->left)+count(root->right);
}

u64 tree_size(Node *root) {
    return count(root) * sizeof(Node);
}

void print_tree(Node *root, int indent) {
    if (root->left == NULL && root->right == NULL) {
        printf("%*" PRIu64 "\n", indent, root->discrete_value);
        return;
    }
    printf("%*c%lf >= feature[%" PRIu64 "]\n", indent, ' ', root->split.value, root->feature);
    print_tree(root->left, indent+2);
    printf("%*c%lf < feature[%" PRIu64 "]\n", indent, ' ', root->split.value, root->feature);
    print_tree(root->right, indent+2);
}

void free_tree(Node *root) {
    if (root == NULL) return;
    free_tree(root->left);
    free_tree(root->right);
    free(root);
}
//~

//~ NOTE(mdizdar): pruning and cutting, not really basic tree functions, but not far off

void prune_helper(Node *root, double base_fitness, Node *current, Data *dataset, u64 depth, u64 classcount[MAX_CLASSES]) {
    if (current->left == NULL && current->right == NULL) {
        ++classcount[current->discrete_value];
        return;
    }
    u64 left[MAX_CLASSES] = {0};
    u64 right[MAX_CLASSES] = {0};
    prune_helper(root, base_fitness, current->left, dataset, depth+1, left);
    prune_helper(root, base_fitness, current->right, dataset, depth+1, right);
    
    bool left_good = false, right_good = false;
    for (u64 i = 0; i < CLASSES; ++i) {
        if (left[i] != 0) {
            left_good = true;
        }
        if (right[i] != 0) {
            right_good = true;
        }
        left[i] += right[i];
    }
    
    if (depth < MIN_DEPTH || !left_good || !right_good) {
        for (u64 i = 0; i < CLASSES; ++i) {
            classcount[i] = 0;
        }
        return;
    }
    Node *tmpleft = current->left, *tmpright = current->right;
    u64 tmpfeature = current->feature;
    current->left = current->right = NULL;
    
    u64 cur = 0;
    current->discrete_value = 0;
    for (u64 i = 1; i < CLASSES; ++i) {
        if (left[i] > cur) {
            current->discrete_value = i;
        }
    }
    double pruned_fitness = FITNESS(root, dataset);
    if (fabs(pruned_fitness - base_fitness) < EPSILON) {
        free_tree(tmpleft); tmpleft = NULL;
        free_tree(tmpright); tmpright = NULL;
        for (u64 i = 0; i < CLASSES; ++i) {
            classcount[i] += left[i];
        }
    } else {
        current->left = tmpleft;
        current->right = tmpright;
        current->feature = tmpfeature;
        for (u64 i = 0; i < CLASSES; ++i) {
            classcount[i] = 0;
        }
    }
}

void cut(Node *root, u64 depth, u64 classcount[MAX_CLASSES]) {
    if (root->left == NULL && root->right == NULL) {
        ++classcount[root->discrete_value];
        return;
    }
    u64 left[MAX_CLASSES] = {0};
    u64 right[MAX_CLASSES] = {0};
    cut(root->left, depth+1, left);
    cut(root->right, depth+1, right);
    for (u64 i = 0; i < CLASSES; ++i) {
        left[i] += right[i];
    }
    if (depth == MAX_DEPTH) {
        free_tree(root->left);
        free_tree(root->right);
        root->left = root->right = NULL;
        u64 cur = 0;
        root->discrete_value = 0;
        for (u64 i = 1; i < CLASSES; ++i) {
            if (left[i] > cur) {
                root->discrete_value = i;
            }
        }
    }
}

static inline void prune(Node *root, Data *dataset) {
    u64 classcount[MAX_CLASSES] = {0};
    if (PRUNE) {
        prune_helper(root, FITNESS(root, dataset), root, dataset, 0, classcount);
    }
    for (u64 i = 0; i < CLASSES; ++i) {
        classcount[i] = 0;
    }
    cut(root, 1, classcount);
}
//~

//~ NOTE(mdizdar): evaluation functions

// traverses the decision tree for a specific row of data
u64 predict(Node *root, Row *instance) {
    if (root->left == NULL && root->right == NULL) {
        return root->discrete_value;
    }
    if (feature_discrete[root->feature]) {
        if (root->split.attribute_index == instance->data[root->feature].attribute_index) {
            return predict(root->right, instance);
        } else {
            return predict(root->left, instance);
        }
    } else {
        if (root->split.value >= instance->data[root->feature].value) {
            return predict(root->right, instance);
        } else {
            return predict(root->left, instance);
        }
    }
}

// categorical crossentropy
double evaluate_crossentropy(Node *member, Data *dataset) {
    double entropy = 0;
    for (u64 i = 0; i < dataset->size; ++i) {
        u64 prediction = predict(member, &dataset->data[i]);
        // categorical crossentropy
        for (u64 j = 0; j < CLASSES; ++j) {
            // NOTE(mdizdar): not 100% sure this is correct, but idk I guess it is
            if (j != prediction) continue;
            if (j == dataset->data[i].label) continue;
            entropy += -LOG2_EPSILON; // uwu
        }
    }
    return entropy / dataset->size; // average categorical crossentropy
}

// accuracy
double evaluate_accuracy(Node *member, Data *dataset) {
    u64 correct = 0;
    for (u64 i = 0; i < dataset->size; ++i) {
        u64 prediction = predict(member, &dataset->data[i]);
        if (prediction == dataset->data[i].label) ++correct;
    }
    return 1.*correct / dataset->size;
}

// calculates F-1 with given true positive, true negative, false positive and false negative counts
static inline double fMeasure(u64 TP, u64 TN, u64 FP, u64 FN) {
    if (TP == 0) return 0; // symbolic value to prevent it from sorting NaNs above everything else
    double precision = 1.*TP/(TP+FP);
    double sensitivity = 1.*TP/(TP+FN);
    return 2.*precision*sensitivity/(precision+sensitivity);
}

// F-1 
double evaluate_fmeasure(Node *member, Data *dataset) {
    u64 TP = 0, TN = 0, FP = 0, FN = 0;
    for (u64 i = 0; i < dataset->size; ++i) {
        u64 prediction = predict(member, &dataset->data[i]);
        if (prediction == 1) {
            if (dataset->data[i].label == 1) {
                ++TP;
            } else {
                ++FP;
            }
        } else {
            if (dataset->data[i].label == 1) {
                ++FN;
            } else {
                ++TN;
            }
        }
    }
    return fMeasure(TP, TN, FP, FN);
}

// I'm not too proud of this function.. it should calculate more stuff, but I couldn't be bothered. it also does predictions on a given dataset
void evaluate_all_and_save(Node *members[], Data *dataset) {
    if (SILENT && OUTPUT_PATH == NULL) {
        return;
    }
    FILE *ofile = OUTPUT_PATH == NULL ? stdout : fopen(OUTPUT_PATH, "w+");
    double avg_fitness = 0;
    for (u64 i = 0; i < INSTANCES; ++i) {
        members[i]->fitness = FITNESS(members[i], dataset);
        avg_fitness += members[i]->fitness;
    }
    sort(members, INSTANCES, SORTING);
    fprintf(ofile, "Best loss: %.10lf\n", members[0]->fitness);
    fprintf(ofile, "Avg. loss: %.10lf\n", avg_fitness / INSTANCES);
    
    fprintf(ofile, "Test set crossentropy: %.10lf\n", evaluate_crossentropy(members[0], dataset));
    fprintf(ofile, "Test set accuracy: %.10lf\n", evaluate_accuracy(members[0], dataset));
    fprintf(ofile, "Test set f-1: %.10lf\n", evaluate_fmeasure(members[0], dataset));
    fprintf(ofile, "Tree depth: %" PRIu64 "\n", depth(members[0]));
    if (OUTPUT_PATH == NULL) {
        return;
    }
    fprintf(ofile, "Predictions: ");
    for (u64 i = 0; i < dataset->size; ++i) {
        fprintf(ofile, "%" PRIu64 " ", predict(members[0], &dataset->data[i]));
    }
    fclose(ofile);
}
//~

//~ NOTE(mdizdar): genetic programming functions

Node *crossover(Node *s1, Node *s2) {
    Node *offspring = copy_tree(s1, NULL);
    Node *n1 = find_random_node(offspring, true, 1./count(offspring));
    Node *n1_parent = n1->parent;
    Node *n2 = copy_tree(find_random_node(s2, true, 1./count(s2)), NULL);
    n2->parent = n1_parent;
    if (n1_parent->left == n1) {
        free_tree(n1_parent->left); n1_parent->left = NULL;
        n1_parent->left = n2;
    } else {
        free_tree(n1_parent->right); n1_parent->right = NULL;
        n1_parent->right = n2;
    }
    return offspring;
}

Node *mutate(Node *s) {
    Node *mutant = copy_tree(s, NULL);
    for (u64 i = 0, lim = rand() % (MAX_MUTATIONS-1) + 1; i < lim; ++i) {
        Node *n1 = find_random_node(mutant, true, 1./count(mutant));
        if (rand() < TURN_TO_LEAF * RAND_MAX) {
            free_tree(n1->left); n1->left = NULL;
            free_tree(n1->right); n1->right = NULL;
            random_leaf(n1);
        } else {
            random_node(n1);
            if (n1->left == NULL && n1->right == NULL) {
                n1->left = create_random_tree(alloc(sizeof(Node)), n1, 0);
                n1->right = create_random_tree(alloc(sizeof(Node)), n1, 0);
            }
        }
    }
    return mutant;
}
//~

//~ NOTE(mdizdar): serialization functions

Node *load_tree(FILE *file, Node * parent) {
    Node *root = alloc(sizeof(Node));
    root->parent = parent;
    char type;
    fscanf(file, " %c", &type);
    if (type == 'l') {
        // leaf
        fscanf(file, " %" SCNu64, &root->discrete_value);
        root->left = NULL;
        root->right = NULL;
    } else {
        // node
        bool was_discrete;
        u64 one_feature = 69;
        fscanf(file, " %" SCNu64 " %" SCNuFAST8, &one_feature, &was_discrete);
        root->feature = one_feature;
        if (was_discrete) {
            fscanf(file, " %" SCNu64, &root->split.attribute_index);
        } else {
            fscanf(file, " %lf", &root->split.value);
        }
        root->left = load_tree(file, root);
        root->right = load_tree(file, root);
    }
    return root;
}

void save_tree(FILE *file, Node *root) {
    if (root->left == NULL && root->right == NULL) {
        fprintf(file, "l %" PRIu64 " ", root->discrete_value);
        return;
    }
    fprintf(file, "n %" PRIu64 " %" PRIuFAST8 " ", root->feature, feature_discrete[root->feature]);
    if (feature_discrete[root->feature]) {
        fprintf(file, "%" PRIu64 " ", root->split.attribute_index);
    } else {
        fprintf(file, "%.10lf ", root->split.value);
    }
    save_tree(file, root->left);
    save_tree(file, root->right);
}

Node **load_instances(const char *filename) {
    FILE *file = fopen(filename, "r");
    u64 prev_classes, prev_features;
    fscanf(file, " %" SCNu64 " %" SCNu64 " %" SCNu64, &INSTANCES, &prev_classes, &prev_features);
    if (CLASSES != prev_classes) {
        fprintf(stderr, "WARNING: the number of classes these trees were trained on (%" PRIu64 ") is different from the number of classes in the provided dataset (%" PRIu64 ")", prev_classes, CLASSES);
    }
    if (FEATURES != prev_features) {
        fprintf(stderr, "WARNING: the number of features these trees were trained on (%" PRIu64 ") is different from the number of features in the provided dataset (%" PRIu64 ")", prev_features, FEATURES);
    }
    Node **members = malloc((INSTANCES + N_CROSSOVER + N_MUTATION) * sizeof(Node *));
    for (u64 i = 0; i < INSTANCES; ++i) {
        members[i] = load_tree(file, NULL);
    }
    fclose(file);
    return members;
}

void save_instances(Node *members[], u64 generation) {
    const u64 len = strlen(CHECKPOINT_PATH) + 100;
    char *output_file = malloc(len * sizeof(char));
    sprintf(output_file, "%scheckpoint_%0*" PRIu64 ".trees", CHECKPOINT_PATH, (int)digits(MAX_GENERATIONS), generation);
    FILE *file = fopen(output_file, "w+");
    fprintf(file, "%" PRIu64 " %" PRIu64 " %" PRIu64 "\n", INSTANCES, CLASSES, FEATURES);
    for (u64 i = 0; i < INSTANCES; ++i) {
        save_tree(file, members[i]);
        fprintf(file, "\n");
    }
    fclose(file);
    free(output_file);
}
//~

//~ NOTE(mdizdar): commandline argument stuff

// prints the help message and exits the program
void help() {
    fprintf(stderr, "-h --help\t\t\t\tdisplays this help text\n");
    fprintf(stderr, "-f --filename [directory]\t\tsets the path to the dataset\n");
    fprintf(stderr, "-s --stratify\t\t\t\tafter shuffling, stratify the dataset\n");
    fprintf(stderr, "-S --seed\t\t\t\tseed for random number generator; 420 by default\n");
    fprintf(stderr, "-t --times [integer]\t\t\thow many times to run the whole process; 1 by default\n");
    fprintf(stderr, "-i --instances [integer]\t\thow many instances there are per generation; 1024 by default\n");
    fprintf(stderr, "-c --crossover [integer]\t\thow many offspring to create each generation; 512 by default\n");
    fprintf(stderr, "-m --mutation [integer]\t\t\thow many mutated clones to create; 512 by default; creates new instances\n");
    fprintf(stderr, "--mutate-to-leaf-ratio [real 0-1]\todds that mutation will transform a node into a leaf; 0.33 by default\n");
    
    fprintf(stderr, "-F --fitness [CE/acc/f1]\t\twhich fitness function to use; F-1 by default\n");
    fprintf(stderr, "\tavailable options are: CE/crossentropy, acc/ACC/accuracy and F1/F-1/f1/f-1\n");
    
    fprintf(stderr, "-p --prune\t\t\t\tprune the trees with reduced error pruning after crossover/mutation\n");
    fprintf(stderr, "-j --jobs [integer]\t\t\tnumber of threads to use\n");
    fprintf(stderr, "-r --ratio --split-ratio [real 0-1]\tsplit ratio for train and test sets; 0.8 by default\n");
    fprintf(stderr, "-g --generations [integer]\t\tnumber of generations to simulate; 3000 by default\n");
    fprintf(stderr, "-d --max-depth [integer]\t\tmaximum depth of each instance; 10 by default\n");
    fprintf(stderr, "-M --max-mutations [integer]\t\tmaximum number of mutations for each instance; 10 by default\n");
    fprintf(stderr, "-C --checkpoints [integer] [directory]\tcreate a checkpoint every N generations; 0 means don't create checkpoints; 0 by default\n");
    fprintf(stderr, "--continue [file]\t\t\tcontinue simulating with instances loaded from file\n");
    fprintf(stderr, "--silent\t\t\t\tdon't print anything to stdout\n");
    fprintf(stderr, "-e --eval-only [file]\t\t\tevaluates the trees stored in the file specified\n");
    fprintf(stderr, "-o --output [file]\t\t\tspecifies output file for final evaluation\n");
    
    exit(0);
}

// parses commandline arguments, doesn't check for any mistakes, anything you fuck up is on you
void parse_args(int argc, char **argv) {
    SORTING = Descending;
    if (argc < 2) {
        help();
    }
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            help();
        } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--filename") == 0) {
            ++i;
            FILENAME = argv[i];
        } else if (strcmp(argv[i], "-s") == 0 || strcmp(argv[i], "--stratify") == 0) {
            STRATIFY = true;
        } else if (strcmp(argv[i], "-S") == 0 || strcmp(argv[i], "--seed") == 0) {
            ++i;
            SEED = atoi(argv[i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--times") == 0) {
            ++i;
            TIMES = atoi(argv[i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--instances") == 0) {
            ++i;
            INSTANCES = atoi(argv[i]);
        } else if (strcmp(argv[i], "-c") == 0 || strcmp(argv[i], "--crossover") == 0) {
            ++i;
            N_CROSSOVER = atoi(argv[i]);
        } else if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--mutation") == 0) {
            ++i;
            N_MUTATION = atoi(argv[i]);
        } else if (strcmp(argv[i], "--mutate-to-leaf-ratio") == 0) {
            ++i;
            TURN_TO_LEAF = atof(argv[i]);
        } else if (strcmp(argv[i], "-F") == 0 || strcmp(argv[i], "--fitness") == 0) {
            ++i;
            if (strcmp(argv[i], "CE") == 0 || strcmp(argv[i], "crossentropy") == 0) {
                FITNESS = evaluate_crossentropy;
                SORTING = Ascending;
            } else if (strcmp(argv[i], "F1") == 0 || strcmp(argv[i], "F-1") == 0 || strcmp(argv[i], "f1") == 0 || strcmp(argv[i], "f-1") == 0) {
                FITNESS = evaluate_fmeasure;
                SORTING = Descending;
            } else if (strcmp(argv[i], "acc") == 0 || strcmp(argv[i], "ACC") == 0 || strcmp(argv[i], "accuracy") == 0) {
                FITNESS = evaluate_accuracy;
                SORTING = Descending;
            }
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prune") == 0) {
            PRUNE = true;
        } else if (strcmp(argv[i], "-j") == 0 || strcmp(argv[i], "--jobs") == 0) {
            ++i;
            NUM_THREADS = atoi(argv[i]);
        } else if (strcmp(argv[i], "-r") == 0 || strcmp(argv[i], "--split-ratio") == 0 || strcmp(argv[i], "--ratio") == 0) {
            ++i;
            SPLIT_RATIO = atof(argv[i]);
        } else if (strcmp(argv[i], "-g") == 0 || strcmp(argv[i], "--generations") == 0) {
            ++i;
            MAX_GENERATIONS = atoi(argv[i]);
        } else if (strcmp(argv[i], "-d") == 0 || strcmp(argv[i], "--max-depth") == 0) {
            ++i;
            MAX_DEPTH = atoi(argv[i]);
        } else if (strcmp(argv[i], "-M") == 0 || strcmp(argv[i], "--max-mutations") == 0) {
            ++i;
            MAX_MUTATIONS = atoi(argv[i]);
        } else if (strcmp(argv[i], "-C") == 0 || strcmp(argv[i], "--checkpoints") == 0) {
            ++i;
            CHECKPOINTS = atoi(argv[i]);
            ++i;
            CHECKPOINT_PATH = argv[i];
        } else if (strcmp(argv[i], "--continue") == 0) {
            ++i;
            CONTINUE_FILE = argv[i];
        } else if (strcmp(argv[i], "--silent") == 0) {
            SILENT = true;
        } else if (strcmp(argv[i], "-e") == 0 || strcmp(argv[i], "--eval-only") == 0) {
            ++i;
            EVAL_ONLY_PATH = argv[i];
        } else if (strcmp(argv[i], "-o") == 0 || strcmp(argv[i], "--output") == 0) {
            ++i;
            OUTPUT_PATH = argv[i];
        }
    }
}
//~

//~ NOTE(mdizdar): THE MAIN FUNCTION, BIG DICK ENERGY ONLY, *** IF YOU HAVE SMALL DICK ENERGY, GO TO JAIL, DO NOT PASS GO, DO NOT COLLECT $200  ***
int main(int argc, char **argv) {
    parse_args(argc, argv);
    
    srand(SEED);
    
    //~ NOTE(mdizdar): reading the dataset
    Data train_data;
    Data test_data;
    Data data;
    
    FILE *f = fopen(FILENAME, "r");
    
    fscanf(f, " %" SCNu64 "%" SCNu64 "%" SCNu64, &data.size, &FEATURES, &CLASSES);
    
    feature_min = alloc(FEATURES * sizeof(double));
    feature_max = alloc(FEATURES * sizeof(double));
    feature_discrete = alloc(FEATURES * sizeof(bool));
    
    for (u64 i = 0; i < FEATURES; ++i) {
        fscanf(f, " %lf%lf%" SCNuFAST8, &feature_min[i], &feature_max[i], &feature_discrete[i]);
    }
    
    data.data = alloc(data.size * sizeof(Row));
    
    for (u64 i = 0; i < data.size; ++i) {
        fscanf(f, " %" SCNu64, &data.data[i].label);
        data.data[i].data = alloc(FEATURES * sizeof(Feature));
        for (u64 j = 0; j < FEATURES; ++j) {
            if (feature_discrete[j]) {
                fscanf(f, " %" SCNu64, &data.data[i].data[j].attribute_index);
            } else {
                fscanf(f, " %lf", &data.data[i].data[j].value);
            }
        }
    }
    fclose(f);
    //~
    
    if (EVAL_ONLY_PATH != NULL) {
        puts("?");
        Node **members = load_instances(EVAL_ONLY_PATH);
        SORT_HELPER = malloc(INSTANCES  * sizeof(Node *));
        evaluate_all_and_save(members, &data);
        return 0;
    }
    
    Node **SAVED = NULL;
    if (CONTINUE_FILE != NULL) {
        SAVED = load_instances(CONTINUE_FILE);
    }
    
    const u64 TOT_INSTANCES = INSTANCES + N_CROSSOVER + N_MUTATION;
    double avg_cross = 0, avg_acc = 0, avg_f1 = 0;
    u64 cnt_f1 = 0;
    
    Node **members = malloc(TOT_INSTANCES * sizeof(Node *));
    SORT_HELPER = malloc(TOT_INSTANCES * sizeof(Node *));
    
    for (u64 times = 0; times < TIMES; ++times) {
        //~ NOTE(mdizdar): preparation
        for (u64 i = 0; i < 10; ++i) {
            shuffle(data.data, data.size, sizeof(Row));
        }
        
        if (STRATIFY) {
            stratify(&data, SPLIT_RATIO);
        }
        train_data.data = data.data;
        train_data.size = (u64)(data.size * SPLIT_RATIO);
        
        test_data.data = data.data + train_data.size;
        test_data.size = data.size - train_data.size;
        
        u64 generation = 0;
        double prev_gen_fitness = HUGE_VAL;
        double gen_fitness = HUGE_VAL;
        double avg_fitness = 0;
        //~
        
        //~ NOTE(mdizdar): gen 0
        for (u64 i = 0; i < INSTANCES; ++i) {
            if (CONTINUE_FILE == NULL) {
                members[i] = create_random_tree(alloc(sizeof(Node)), NULL, rand()%(MAX_DEPTH-1)+2);
            } else {
                members[i] = copy_tree(SAVED[i], NULL);
            }
            prune(members[i], &train_data);
            members[i]->fitness = FITNESS(members[i], &train_data);
        }
        for (u64 i = INSTANCES; i < TOT_INSTANCES; ++i) {
            members[i] = create_random_tree(alloc(sizeof(Node)), NULL, rand()%(MAX_DEPTH-1)+2);
            prune(members[i], &train_data);
            members[i]->fitness = FITNESS(members[i], &train_data);
        }
        //~
        
        //~ NOTE(mdizdar): genetic programming starts here
        do {
            sort(members, TOT_INSTANCES, SORTING);
            prev_gen_fitness = gen_fitness;
            gen_fitness = members[0]->fitness;
            
            for (u64 i = 0; i < INSTANCES; ++i) {
                avg_fitness += members[i]->fitness;
            }
            
            
            if (!SILENT && (CHECKPOINTS == 0 || generation % CHECKPOINTS == 0)) {
                printf("Generation %" PRIu64 ": Best %.10lf | Avg %.10lf\n", generation, gen_fitness, avg_fitness / INSTANCES);
            }
            if (CHECKPOINTS > 0 && generation > 0 && generation % CHECKPOINTS == 0) {
                save_instances(members, generation);
            }
            avg_fitness = 0;
            // crossover
            for (u64 i = INSTANCES; i < INSTANCES+N_CROSSOVER; ++i) {
                u64 s1 = rand() % INSTANCES;
                u64 s2 = rand() % INSTANCES;
                while (s2 == s1) s2 = rand() % INSTANCES;
                free_tree(members[i]);
                members[i] = crossover(members[s1], members[s2]);
            }
            
            // mutate
            for (u64 i = INSTANCES+N_CROSSOVER; i < TOT_INSTANCES; ++i) {
                u64 s = rand() % (INSTANCES+N_CROSSOVER);
                free_tree(members[i]);
                members[i] = mutate(members[s]);
            }
            
#pragma omp parallel num_threads(NUM_THREADS)
            {
                s64 i;
#pragma omp for
                for (i = INSTANCES; i < (s64)TOT_INSTANCES; ++i) {
                    prune(members[i], &train_data);
                    members[i]->fitness = FITNESS(members[i], &train_data);
                }
            }
        } while (generation++ < MAX_GENERATIONS);
        //~
        
        //~ NOTE(mdizdar): some extra stats I guess
        double cross = evaluate_crossentropy(members[0], &test_data);
        double acc = evaluate_accuracy(members[0], &test_data);
        double f1 = evaluate_fmeasure(members[0], &test_data);
        
        if (TIMES == times+1) {
            evaluate_all_and_save(members, &test_data);
        } else {
            printf("Train set loss: %.10lf\n", FITNESS(members[0], &train_data));
            printf("Test set crossentropy: %.10lf\n", cross);
            printf("Test set accuracy: %.10lf\n", acc);
            printf("Test set f-1: %.10lf\n", f1);
            printf("Tree depth: %" PRIu64 "\n", depth(members[0]));
        }
        
        avg_cross += cross;
        avg_acc += acc;
        if (f1 > 0) {
            avg_f1 += f1;
            ++cnt_f1;
        }
        //~
        
        for (u64 i = 0; i < TOT_INSTANCES; ++i) {
            free_tree(members[i]); // freeing memory in 2021? I sure hope not
        }
    }
    printf("\nAverage crossentropy: %.10lf\n", avg_cross / TIMES);
    printf("Average accuracy: %.10lf\n", avg_acc / TIMES);
    if (cnt_f1 > 0) avg_f1 /= cnt_f1;
    printf("Average f-1: %.10lf (over %" PRIu64 " non NaN runs)\n", avg_f1, cnt_f1);
    
    return 0;
}