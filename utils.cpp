// Include librarys
#include <string>
#include <vector>
#include <vector>
#include <iostream>
#include <fstream>
#include <opencv2/core/core.hpp>

// Include header
#include "utils.h"

using namespace std;
using namespace cv;

// for string delimiter
vector<string> split(string s, string delimiter) {
	size_t pos_start = 0, pos_end, delim_len = delimiter.length();
	string token;
	vector<string> res;
    //Encontramos el triangulo asociado a cada defecto covexo para determinar el angulo
	while ((pos_end = s.find(delimiter, pos_start)) != string::npos) {
		token = s.substr(pos_start, pos_end - pos_start);
		pos_start = pos_end + delim_len;
		res.push_back(token);
	}

	res.push_back(s.substr(pos_start));
	return res;
}

double euclideanDistance(double * m1, double * m2) {
	double suma = 0.0;
	for (int i = 0; i < 7; i++) {
		suma += (m1[i] - m2[i]) * (m1[i] - m2[i]);
	}
	return sqrt(suma);
	// norm(m1, m2, NORM_L2);
}

double logTransform(double momento) {
	return -1 * copysign(1.0, momento) * log10(abs(momento));
}

string matchDescriptores(double huMomentsNew[7], string fileName, double valueAcept) {
	double huMomentsOld[7];
	string line, gesture = NEW_GESTURE;
	vector<string> atributes;
	ifstream infile(fileName);
	if (infile.is_open()) {
		while (getline(infile, line))
		{
			// get all atributes form line
			atributes = split(line, ";");
			for (int i = 0; i < 7; i++)
			{
				huMomentsOld[i] = stod(atributes[i]);
			}
			// compare two moments
			double distancia = euclideanDistance(huMomentsNew, huMomentsOld);
			cout << "Distancia: " << distancia << ";  ";
			if (distancia < valueAcept) {
				gesture = atributes[7];
			}
		}
		cout << endl;
		// close file stream
		infile.close();
	}
	return gesture;
}

bool saveDescriptores(double huMomentsNew[7], string fileName, double valueAcept) {
	double huMomentsOld[7];
	int next = 1;
	string line, gesture;
	vector<string> atributes;
	ifstream infile(fileName);
	if (infile.is_open()) {
		gesture = "";
		while (getline(infile, line))
		{
			next++;
			// get all atributes form line
			gesture += line + "\n";
			atributes = split(line, ";");
			for (int i = 0; i < 7; i++)
			{
				huMomentsOld[i] = stod(atributes[i]);
			}
			// compare two moments
			if (euclideanDistance(huMomentsNew, huMomentsOld) < valueAcept) {
				return false;
			}
		}
		infile.close();
		ofstream oufile(fileName);
		if (oufile.is_open()) {
			for (int i = 0; i < 7; i++)
			{
				gesture += to_string(huMomentsNew[i]) + ";";
			}
			gesture += "Gesto(" + to_string(next) + ")";
			oufile << gesture << endl;
			oufile.close();
		}
	}
	return true;
}