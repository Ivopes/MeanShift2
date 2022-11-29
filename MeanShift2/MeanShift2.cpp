#include <iostream>
#include <vector>
#define M_PI 3.14159265358979323846
#include <cmath>
#include <cstring>
#include <omp.h>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <string>
#include <sstream>
#include <regex>

using namespace std;

int dims;
int totalCount;
double* vectors;
double** vectorsPoints;
double* centroids;
double** centroidsPoints;
double windowSize;

void LoadData(string filepath);
void NormalizeDataset();
double FindMin(int index);
double FindMax(int index);
void Run(double windowS);
double* Kernel(double* input);
double* Hustota();
double* Add(double* one, double* two);

int main(int argc, char* argv[])
{
	//LoadData("C:\\Users\\hapes\\Downloads\\meanSoubory\\mnist_test2.csv");
	LoadData("C:\\Users\\hapes\\Downloads\\vektory.txt");

	// normalizuju data
	NormalizeDataset();

	//vypocet
	Run(5);

	return 0;
}

void Run(double windowS) {
	
	windowSize = windowS;

#pragma omp parallel for
	for (int i = 0; i < totalCount; i++)
	{
		double* centroid = centroidsPoints[i];
		double** okoli = (double**)malloc(totalCount * sizeof(double*));
		for (int j = 0; j < totalCount; j++) okoli[j] = nullptr;

		//najit okoli bodu
		int countIndex = -1;
		for (int j = 0; j < totalCount; j++)
		{
			//if (j == i) continue;
			double* vector = vectorsPoints[j];
			double sum = 0;

			for (int k = 0; k < dims; k++)
			{
				sum += (vector[k] - centroid[k]) * (vector[k] - centroid[k]);
			}
			
			if (sum < (windowSize*windowSize) && sum != 0)
			{
				//pridat do okoli bodu
				okoli[++countIndex] = vectorsPoints[j];
			}
		}
		
		//vypocitat posun a posunout
		//double* posun = Hustota(i, okoli, countIndex + 1);

		//Add(centroid, posun);

	}
	
}
double* Hustota(int pos, double** okoliBodu, int pocet) {

	//vypoctu horni cast funkce	
	double* horni = 0;
	/*
	for (int i = 0; i < pocet; i++)
	{
		double* vTemp = (double*)malloc(dims * sizeof(double));
		double* v = okoliBodu[i];
		for (int j = 0; j < dims; i++)
		{
			vTemp[j] = v[j] - centroids[pos * dims + j];
		}
		//Kernel(vTemp);
	}
	*/

	//vypoctu dolni cast funkce	
	//naplnim nulama SUM
	double* dolni = (double*)malloc(dims * sizeof(double)); 
	for (int i = 0; i < dims; i++) dolni[i] = 0;

	double* vTemp = (double*)malloc(dims * sizeof(double));
	for (int i = 0; i < pocet; i++)
	{
		double* v = okoliBodu[i];
		for (int j = 0; j < dims; i++)
		{
			vTemp[j] = v[j] - centroids[pos * dims + j];
		}
		dolni = Add(dolni, Kernel(vTemp));
	}

	//podelit hodni/dolni

	//vratit vektor posunu
	return nullptr;
}
double* Kernel(double* input) {
	double wPwr = windowSize * windowSize;
	double* vTemp = (double*)malloc(dims * sizeof(double));

	for (int i = 0; i < dims; i++)
	{
		//double vHelp = vectors[pos * dims + i];
		double vHelp = input[i];
		vTemp[i] = exp(-(vHelp) / (2 * wPwr));
	}
	
	return vTemp;
}
double* Add(double* one, double* two) {

	for (int i = 0; i < dims; i++)
	{
		one[i] += two[i];
	}
	return one;
}
void LoadData(string filepath) {
	
	std::ifstream file(filepath);

	if (file.is_open())
	{
		std::string line;
		int pos;
		dims = 0;
		vector<double> v;

		std::getline(file, line);
		std::stringstream sin(line);
		char c;
		//ignore prvni radek, pouze pocitame dimenze
		while (!sin.eof()) {
			sin >> c;
			if (c == 'x')
				dims++;
		}

		int p = 0;
		while (std::getline(file, line)) {
			p++;
			//line = std::regex_replace(line, std::regex(","), " ");
			stringstream sin(line);

			//ignore prvni cislo
			sin >> pos >> c;

			while (!sin.eof()) {
				sin >> pos >> c;
				v.push_back(pos);
			}
		}

		file.close();

		totalCount = v.size() / dims;
		size_t mem_size = totalCount * dims * sizeof(double);
		vectors = (double*)malloc(mem_size);
		memcpy(vectors, &(v[0]), mem_size);
		centroids = (double*)malloc(mem_size);
		memcpy(centroids, &(vectors[0]), mem_size);
		vectorsPoints = (double**)malloc(totalCount * sizeof(double*));
		centroidsPoints = (double**)malloc(totalCount * sizeof(double*));

		for (int i = 0; i < totalCount; i++)
		{
			vectorsPoints[i] = &vectors[i * dims];
			centroidsPoints[i] = &centroids[i*dims];
		}
	}
}
void NormalizeDataset() {
	double* min_vec;
	min_vec = (double*)malloc(dims * sizeof(double));
	double* max_vec;
	max_vec = (double*)malloc(dims * sizeof(double));

#pragma omp parallel for
	for (int i = 0; i < dims; i++)
	{
		min_vec[i] = FindMin(i);
		max_vec[i] = FindMax(i);
	}

#pragma omp parallel for
	for (int i = 0; i < totalCount; i++)
	{
		for (int j = 0; j < dims; j++)
		{
			int pos = i * dims + j;
			vectors[pos] = (vectors[pos] - min_vec[j]) / (max_vec[j] - min_vec[j]);
			if (vectors[pos] != vectors[pos]) vectors[pos] = 0;
			centroids[pos] = vectors[pos];
		}
	}
}
double FindMin(int index) 
{
	double min_val = vectors[0];
#pragma omp simd reduction(min:min_val)
	for (int i = 0; i < totalCount; i++)
	{
		min_val = min_val < vectors[i * dims + index] ? min_val : vectors[i * dims + index];
	}

	return min_val;
}
double FindMax(int index)
{
	double max_val = vectors[0];
#pragma omp simd reduction(max:max_val)
	for (int i = 0; i < totalCount; i++)
	{
		max_val = max_val > vectors[i * dims + index] ? max_val : vectors[i * dims + index];
	}

	return max_val;
}

