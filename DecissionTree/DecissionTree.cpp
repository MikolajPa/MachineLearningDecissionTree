#include <iostream>
#include<vector>
#include <limits>

using namespace std;

vector<int> testValues;
vector<bool> results;
const int NUMBER_OF_FEATURES = 1;


    struct Node {
        int data;
        Node* noNode;
        Node* rightNode;
        double threeshold = numeric_limits<double>::lowest();
        string feature;

        Node(int threeshold) {
            this->threeshold = threeshold;
            this->noNode = nullptr;
            this->rightNode = nullptr;
        }

        bool isLeave()
        {
            return threeshold == numeric_limits<double>::lowest() ? true : false;
        }
    };

    class MainTree {
    public:
        Node* root;
        double maxDepth;
        double minSampleSplit;
        double nClassLabels;
        double nSamples;
        double nFeatures;
        MainTree(double _maxDepth, double _minSamples) {
            root = nullptr;
            maxDepth = _maxDepth;
            _minSamples = minSampleSplit;
        }

        Node* getRoot()
        {
            return root;
        }
       
        void addNode(int threehold, Node& parent, bool no) {
            Node* newNode = new Node(threehold);

            if (root == nullptr) {
                root = newNode;
            }
            else 
            {
                if (no) 
                {
                    
                    parent.noNode = new Node(threehold);
                    cout << threehold;
                }
                else 
                {
                    parent.rightNode = new Node(threehold);
                    cout << threehold;
                }

            }
        }

        bool is_finished(int depth)
        {
            if (depth >= maxDepth || nClassLabels == 1 || nSamples < minSampleSplit)
                return true;
            return false;
        }

        void buildTree(int n_samples, int n_features)
        {
            nSamples = n_samples;
            nFeatures = n_features;



                
        }

        void preOrderTraversal(Node* focusNode) {
            if (focusNode != nullptr) {
                std::cout << focusNode->threeshold << " ";
                preOrderTraversal(focusNode->noNode);
                preOrderTraversal(focusNode->rightNode);
            }
        }
    };



int main()
{
#pragma region DATA_INIT
    testValues.push_back(1);
    testValues.push_back(2);
    testValues.push_back(3);
    testValues.push_back(5);
    testValues.push_back(6);
    testValues.push_back(7);
    results.push_back(1);
    results.push_back(1);
    results.push_back(1);
    results.push_back(0);
    results.push_back(0);
    results.push_back(0);
#pragma endregion

    MainTree tree(2);
    Node newNode = Node(1);

    tree.addNode(1, newNode, false);
    tree.addNode(2, newNode, true);

    tree.preOrderTraversal(tree.getRoot());

    cout << "Hello World!\n";
}

