#pragma once
#include <vector>
#include <opencv2/core/core.hpp>

using namespace std;
using namespace cv;

const string NEW_GESTURE = "Nuevo Gesto (+)";

// for string delimiter
vector<string> split(string s, string delimiter);

// for calc euclidean distance
double euclideanDistance(double m1[7], double m2[7]);

double logTransform(double momento);

bool saveDescriptores(double huMomentsNew[7], string fileName, double valueAcept);

string matchDescriptores(double huMomentsNew[7], string fileName, double valueAcept);