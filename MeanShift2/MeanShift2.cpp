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


const double MOVE_THRESHOLD = 0.0000001;
int dims;
int totalCount;
double* vectors;
double** vectorsPoints;
double* centroids;
double** centroidsPoints;
double windowSize;
bool* settled;

void LoadData(string filepath);
void LoadDataTest(string filepath);
void NormalizeDataset();
double FindMin(int index);
double FindMax(int index);
void Run(double windowS);
double Kernel(double input);
bool Hustota(double** okoliBodu, int pocet, double* centroid);
bool PosunHustotaTest(int pos, double** okoliBodu, int pocet, double* centroid);
double* Add(double* one, double* two);
double EuclidDistance(double* one, double* two);

int main(int argc, char* argv[])
{
	//LoadData("C:\\Users\\hapes\\Downloads\\meanSoubory\\mnist_test2.csv");
	//LoadData("C:\\Users\\hapes\\Downloads\\vektory.txt");
	LoadDataTest("C:\\Users\\hapes\\Downloads\\data_dim_txt\\dim2.txt");

	// normalizuju data
	NormalizeDataset();

	//vypocet
	Run(.1);

	return 0;
}

void Run(double windowS) {
	
	windowSize = windowS;
	int running = totalCount;
	//omp_set_num_threads(8);


	while (running > 0)
	{

#pragma omp parallel for
		for (int i = 0; i < totalCount; i++)
		{
			if (settled[i]) continue;
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
			//bool posunul = PosunHustotaTest(i, okoli, countIndex + 1, centroid);
			bool posunul = Hustota(okoli, countIndex + 1, centroid);

			free(okoli);
	#pragma omp critical
			if (!posunul)
			{
				settled[i] = true;
				running--;
			}
		}
	}
	
}
bool PosunHustotaTest(int pos, double** okoliBodu, int pocet, double* centroid) {

	//vypoctu normalniho centroidu
	double* sum = (double*)malloc(dims * sizeof(double));
#pragma omp parallel for
	for (int i = 0; i < dims; i++) sum[i] = 0;
#pragma omp parallel for
	for (int i = 0; i < pocet; i++)
	{
		double* vektor = okoliBodu[i];
		Add(sum, vektor);
	}
	bool posunulSe = false;
	for (int i = 0; i < dims; i++)
	{
		sum[i] /= pocet;

		if (sum[i] != centroid[i]) posunulSe = true;
	}

	//Add(centroid, sum);
	memcpy(centroid, sum, dims * sizeof(double));

	free(sum);

	return posunulSe;
}
bool Hustota(double** okoliBodu, int pocet, double* centroid) {

	double* horni = (double*)malloc(dims * sizeof(double));
	double** tempVysledky = (double**)malloc(pocet * sizeof(double*));
	
	//vypoctu horni cast zlomku	
#pragma omp parallel for
	for (int i = 0; i < pocet; i++)
	{
		double* temp = (double*)malloc(dims * sizeof(double));
		double* v = okoliBodu[i];
		memcpy(temp, v, dims * (sizeof(double)));
		double distance = EuclidDistance(centroid, v);
		double k = Kernel(distance);
		for (int j = 0; j < dims; j++) temp[j] *= k;

		tempVysledky[i] = temp;
	}

	//Paralleni SUM horni casti
#pragma omp parallel for
	for (int i = 0; i < dims; i++)
	{
		horni[i] = 0;
		for (int j = 0; j < pocet; j++)
		{
			double* v = tempVysledky[j];
			horni[i] += v[i];
		}
	}

	//vypoctu dolni cast funkce	
	double dolni = 0; 

#pragma omp simd reduction(+:dolni)
	for (int i = 0; i < pocet; i++)
	{
		double* v = okoliBodu[i];
		double distance = EuclidDistance(centroid, v);
		double k = Kernel(distance);

		dolni += k;
	}

	bool didMove = false;

	//podelit hodni/dolni
#pragma omp parallel for
	for (int i = 0; i < dims; i++)
	{
		double t = horni[i] / dolni;
		if (t != centroid[i]) didMove = true;
		centroid[i] = t;
		//horni[i] = t;
	}

	//uvolnit pamet
#pragma omp parallel for
	for (int i = 0; i < pocet; i++)
	{
		free(tempVysledky[i]);
	}
	free(tempVysledky);
	free(horni);

	//vratit vektor posunu
	return didMove;
}
double Kernel(double input) {
	double wPwr = windowSize * windowSize;

	double res = exp(-(input/(wPwr*2)));
	
	return res;
}
double EuclidDistance(double* one, double* two)
{
	double sum = 0;
	
#pragma omp parallel for
	for (int i = 0; i < dims; i++)
	{
		sum += (one[i] - two[i]) * (one[i] - two[i]);
	}

	return sqrt(sum);
}
double* Add(double* one, double* two) {

	for (int i = 0; i < dims; i++)
	{
		one[i] += two[i];
	}
	return one;
}
void LoadDataTest(string filepath) {

	std::ifstream file(filepath);

	if (file.is_open())
	{
		std::string line;
		int pos;
		dims = 0;
		vector<double> v;

		dims = 2;
		int p = 0;
		while (std::getline(file, line)) {
			p++;
			stringstream sin(line);

			while (!sin.eof()) {
				sin >> pos;
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
		settled = (bool*)malloc(totalCount * sizeof(bool));

		for (int i = 0; i < totalCount; i++)
		{
			vectorsPoints[i] = &vectors[i * dims];
			centroidsPoints[i] = &centroids[i * dims];
			settled[i] = false;
		}

	}
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
		settled = (bool*)malloc(totalCount * sizeof(bool));

		for (int i = 0; i < totalCount; i++)
		{
			vectorsPoints[i] = &vectors[i * dims];
			centroidsPoints[i] = &centroids[i*dims];
			settled[i] = false;
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

