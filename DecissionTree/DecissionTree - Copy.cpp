#include <iostream>
#include<vector>
#include <limits>
#include <map>
#include <set>
#include <math.h>
#include <utility>
#include <string>
#include <algorithm>
#include <unordered_map>
#include <random>
#include<numeric>
#include <sstream>
#include <fstream>


using namespace std;

vector<double> testValues;
vector<bool> results;
const int NUMBER_OF_FEATURES = 1;

static vector<string> GetVectorSplitted(vector<string> vectorToSplit, int startIndex, int endIndex)
{
    vector<string> newVector;
    for (int i = startIndex;i < endIndex;i++)
    {
        newVector.push_back(vectorToSplit[i]);
    }
    return newVector;
}

vector<double> getColumn(const vector<vector<double>>& X, int i) {
    vector<double> column(X.size());
    transform(X.begin(), X.end(), column.begin(), [i](const auto& row) {
        return row[i];
        });
    return column;
}

static vector<double> GetVectorSplitted(vector<double> vectorToSplit, int startIndex, int endIndex)
{
    vector<double> newVector;
    for (int i = startIndex;i < endIndex;i++)
    {
        newVector.push_back(vectorToSplit[i]);
    }
    return newVector;
}

static string mostCommonLabel(vector<string> labels)
{
    map<string, int> dictionary;
    for (auto label : labels)
    {
        if (dictionary.count(label) == 0)
        {
            dictionary[label] = 1;
        }
        else
        {
            dictionary[label] ++;
        }
    }

    auto pr = max_element(dictionary.begin(), dictionary.end(), [](const auto& x, const auto& y) {
        return x.second < y.second;
        });

    return pr->first;
}


vector<int> randomChoice(int n, int k) {
    vector<int> indices(n);
    iota(indices.begin(), indices.end(), 0);
    shuffle(indices.begin(), indices.end(), default_random_engine());
    indices.resize(k);
    return indices;
}

double calculateEntropy(vector<string> y) {
    int n = y.size();
    unordered_map<string, int> cnt;
    for (int i = 0; i < n; i++) {
        cnt[y[i]]++;
    }
    double entropy = 0.0;
    for (auto it : cnt) {
        double p = (double)it.second / n;
        entropy -= p * log2(p);
    }
    return entropy;
}

//static double calculateEntropy(vector<string> labels)
//{
//    map<string, int> dictionary;
//    for (auto label : labels)
//    {
//        if (dictionary.count(label) == 0)
//        {
//            dictionary[label] = 1;
//        }
//        else
//        {
//            dictionary[label] ++;
//        }
//    }
//
//    double entropy = 0;
//
//    for (const auto& pair : dictionary)
//    {
//        double firstPart = static_cast<double>(pair.second) / static_cast<double>(labels.size());
//        double secondPart = log(static_cast<double>(pair.second) / static_cast<double>(labels.size()));
//        double score = firstPart * secondPart;
//        entropy += score;
//    }
//
//    return entropy;
//}

static vector<string> getUniqueValues(vector<string> vectorToRemoveDuplicates)
{
    set<string> s;
    unsigned size = vectorToRemoveDuplicates.size();
    for (unsigned i = 0; i < size; ++i)
        s.insert(vectorToRemoveDuplicates[i]);
    vectorToRemoveDuplicates.assign(s.begin(), s.end());

    return vectorToRemoveDuplicates;
}

vector<string> subvector(const vector<string>& vec, const vector<int>& idx) 
{
    vector<string> subvec(idx.size());
    for (int i = 0; i < idx.size(); i++) {
        subvec[i] = vec[idx[i]];
    }
    return subvec;
}

vector<vector<double>> subsetRows(const vector<vector<double>>& X, const vector<int>& indices) 
{
    vector<vector<double>> subset(indices.size());
    transform(indices.begin(), indices.end(), subset.begin(), [&X](int i) 
        {
        return X[i];
        });
    return subset;
}

vector<string> subsetRows(const vector<string>& X, const vector<int>& indices) {
    vector<string> subset(indices.size());
    transform(indices.begin(), indices.end(), subset.begin(), [&X](int i) {
        return X[i];
        });
    return subset;
}


class Node {
public:
    int feature;
    double threshold;
    Node* left;
    Node* right;
    string value;

    Node(int feature_ = 0, double threshold_ = 0.0, Node* left_ = nullptr, Node* right_ = nullptr, string value_ = "") :
        feature(feature_), threshold(threshold_), left(left_), right(right_), value(value_) {}

    bool is_leaf() const {
        return !value.empty();
    }
};

struct Split {
    double score = -1;
    int feat = -1;
    double thresh = 0.0;
};

class MainTree {
    public:
        Node* root = nullptr;
        double maxDepth;
        double minSampleSplit;
        double nClassLabels;
        double nSamples;
        double nFeatures;
        string mostCL;
        vector<string> classLabels;
        MainTree(double _maxDepth, double _minSamples) {
            root = nullptr;
            maxDepth = _maxDepth;
            minSampleSplit = _minSamples;
        }

        Node* getRoot()
        {
            return root;
        }
        //https://towardsdatascience.com/implementing-a-decision-tree-from-scratch-f5358ff9c4bb

        bool is_finished(int depth)
        {
            if (depth >= maxDepth || nClassLabels == 1 || nSamples < minSampleSplit)
                return true;
            return false;
        }

        struct retVals {
            int leftVals, rightVals;
        };

        pair<vector<int>, vector<int>> create_split(const vector<double>& X, double thresh) {
            vector<int> left_idx;
            vector<int> right_idx;
            for (int i = 0; i < X.size(); i++) {
                if (X[i] <= thresh) {
                    left_idx.push_back(i);
                }
                else {
                    right_idx.push_back(i);
                }
            }
            return make_pair(left_idx, right_idx);
        }

        double information_gain(const vector<double>& X, const vector<string>& y, double thresh) 
        {
            double parent_loss = calculateEntropy(y);
            auto split = create_split(X, thresh);
            vector<int> left_idx = split.first;
            vector<int> right_idx = split.second;
            int n = y.size();
            int n_left = left_idx.size();
            int n_right = right_idx.size();

            if (n_left == 0 || n_right == 0) {
                return 0;
            }

            double child_loss = (n_left / static_cast<double>(n)) * calculateEntropy(subvector(y, left_idx)) + (n_right / static_cast<double>(n)) * calculateEntropy(subvector(y, right_idx));
            return parent_loss - child_loss;
        }

        //pair<int, double> best_split(vector<vector<double>> X, vector<string> y, vector<int> features) 
        //{
        //    pair<int, double> split = make_pair(-1, 0.0);

        //    for (int feat : features) 
        //    {
        //        vector<double> X_feat;
        //        for (const auto& row : X) 
        //        {
        //            X_feat.push_back(row[feat]);
        //        }

        //        set<double> thresholds(X_feat.begin(), X_feat.end());
        //        for (double thresh : thresholds) 
        //        {
        //            double score = information_gain(X_feat, y, thresh);
        //            if (score > split.second) {
        //                split.first = feat;
        //                split.second = score;
        //            }
        //        }
        //    }

        //    return split; //feat thresh
        //}

        Split best_split(std::vector<std::vector<double>> X, std::vector<string> y, std::vector<int> features) {
            Split split;

            for (auto& feat : features) {
                std::vector<double> X_feat;
                for (auto& row : X) {
                    X_feat.push_back(row[feat]);
                }
                std::sort(X_feat.begin(), X_feat.end());
                std::vector<double> thresholds;
                for (int i = 0; i < X_feat.size() - 1; i++) {
                    if (X_feat[i] != X_feat[i + 1]) {
                        thresholds.push_back((X_feat[i] + X_feat[i + 1]) / 2.0);
                    }
                }
                for (auto& thresh : thresholds) {
                    double score = information_gain(X_feat, y, thresh);

                    if (score > split.score) {
                        split.score = score;
                        split.feat = feat;
                        split.thresh = thresh;
                    }
                }
            }

            return split;
        }

        Node* build_tree(const vector<vector<double>>& X, const vector<string>& y, int depth) 
        {
            nFeatures = X[0].size();
            nClassLabels = getUniqueValues(y).size();
            nSamples = X.size();

            if (is_finished(depth)) 
            {
                string most_common_label = mostCommonLabel(y);
                return new Node(0,0,nullptr,nullptr,most_common_label);
            }

            // get best split
            vector<int> rnd_feats = randomChoice(nFeatures, nFeatures);
            Split best_split_pair = best_split(X, y, rnd_feats);
            int best_feat = best_split_pair.feat;
            double best_thresh = best_split_pair.thresh;

            // grow children recursively
            pair<vector<int>, vector<int>> split_pair = create_split(getColumn(X, best_feat), best_thresh);
            vector<int> left_idx = split_pair.first;
            vector<int> right_idx = split_pair.second;
            Node* left_child = build_tree(subsetRows(X, left_idx), subsetRows(y, left_idx), depth + 1);
            Node* right_child = build_tree(subsetRows(X, right_idx), subsetRows(y, right_idx), depth + 1);
            return new Node(best_feat, best_thresh, left_child, right_child);
        }

        string traverse_tree(const vector<double>& x, Node* node) 
        {
            if (node->is_leaf()) {
                return node->value;
            }

            if (x[node->feature] <= node->threshold) {
                return traverse_tree(x, node->left);
            }

            return traverse_tree(x, node->right);
        }

        vector<string> predict(vector<vector<double>> X, vector<string> yreal) 
        {
            int good=0, bad=0;
            vector<string> predictions;
            int i = 0;
            for (auto& x : X) 
            {
                string ypred = traverse_tree(x, root);
                if (ypred != yreal[i])
                    bad++;
                else
                {
                    good++;
                }
                string xd = yreal[i];
                predictions.push_back(ypred);
                i++;
            }
            return predictions;
        }

        void fit(vector<vector<double>>& X, vector<string>& y) 
        {
            root = build_tree(X, y, 0);
        }
    };



    double accuracy(vector<string>& y_true, vector<string>& y_pred) {
        int correct_predictions = 0;
        int num_samples = y_true.size();

        for (int i = 0; i < num_samples; i++) {
            if (y_true[i] == y_pred[i]) {
                correct_predictions++;
            }
        }

        return static_cast<double>(correct_predictions) / num_samples;
    }

    int main() {
        string filename = "breastcancer.csv";
        ifstream file(filename);

        if (!file.is_open()) {
            cout << "Failed to open file: " << filename << endl;
            return 1;
        }

        ifstream infile("breastcancer.csv");
        string line;
        getline(infile, line);

        // Read rest of file as data
        vector<vector<double>> X;
        vector<string> y;
        while (getline(infile, line)) {
            istringstream iss(line);
            vector<double> row;
            string value;

            while (getline(iss, value, ',')) {
                if (iss.eof()) break; // last value is label
                row.push_back(stod(value));
            }

            X.push_back(row);
            y.push_back(value);
        }

        vector<vector<double>> X_train, X_test;
        vector<string> y_train, y_test;

        random_device rd;
        mt19937 g(rd());
        shuffle(X.begin(), X.end(), g);
        shuffle(y.begin(), y.end(), g);

        int num_train_samples = static_cast<int>(X.size() * 0.8);

        y_train = y;
        y_test = y;

        for (int i = 0; i < num_train_samples; i++) {
            X_train.push_back(X[i]);

        }

        for (int i = num_train_samples; i < X.size(); i++) {
            X_test.push_back(X[i]);
        }

        MainTree model(30,2);
        model.fit(X_train, y_train);

        // Make predictions on the test set
        vector<string> y_pred = model.predict(X_train, y_train);

        // Calculate accuracy of the model
        //double acc = accuracy(y_test, y_pred);
        //cout << "Accuracy: " << acc << endl;

        return 0;
    }
