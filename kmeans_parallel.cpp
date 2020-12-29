// Implementation of the KMeans Algorithm
// reference: http://mnemstudio.org/clustering-k-means-example-1.htm

#include <iostream>
#include <vector>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <exception>
#include <unordered_set>
#include <random>

#include "tbb/tbb.h"
#include "tbb/blocked_range.h"
#include "tbb/parallel_for.h"
#include "tbb/task_scheduler_init.h"

using namespace std;

class Point {
private:
	int id_point, id_cluster;
	vector<double> values ;
	int total_values;
	string name;

public:
	Point(int id_point, vector<double>& values, string name = "") {
		this->id_point = id_point;
		total_values = values.size();

		for(int i = 0; i < total_values; i++)
			this->values.push_back(values[i]);

		this->name = name;
		id_cluster = -1;
	}

    double distance(const Point endpoint) {
        double square_total = 0;
        size_t i = 0;
        while (true) {
            __m256d vec1 = { this->values[i], this->values[i + 1], this->values[i + 2], this->values[i + 3] };
            __m256d vec2 = { endpoint.values[i], endpoint.values[i + 1], endpoint.values[i + 2], endpoint.values[i + 3] };
            __m256d diff = _mm256_sub_pd(vec1, vec2);
            __m256d total = _mm256_mul_pd(diff, diff);
            square_total += total[0];
            square_total += total[1];
            square_total += total[2];
            square_total += total[3];
            if (i + 4 < total_values) i += 4;
            else break;
        }
        for (; i < total_values; ++i) {
            double diff = this->values[i] - endpoint.values[i];
            square_total += diff * diff;
        }
        return sqrt(square_total);
    }

	int getID() {
		return id_point;
	}

	void setCluster(int id_cluster) {
		this->id_cluster = id_cluster;
	}

	int getCluster() {
		return id_cluster;
	}

	double getValue(int index) {
		return values[index];
	}

	int getTotalValues() {
		return total_values;
	}

	void addValue(double value) {
		values.push_back(value);
	}

	string getName() {
		return name;
	}
};

class Cluster {
private:
	int id_cluster;
	vector<double> central_values;
	vector<Point> points;

public:
	Cluster(int id_cluster, Point point) {
		this->id_cluster = id_cluster;
		int total_values = point.getTotalValues();
		for(int i = 0; i < total_values; i++)
			central_values.push_back(point.getValue(i));

		points.push_back(point);
	}

    double distance(Point endpoint) {
        int total_values = this->central_values.size();
        double square_total = 0;
        for (int i = 0; i < total_values; ++i) {
            double diff = this->central_values[i] - endpoint.getValue(i);
            square_total += diff * diff;
        }
        return sqrt(square_total);
    }

	void addPoint(Point point) {
		points.push_back(point);
	}

	bool removePoint(int id_point) {
		int total_points = points.size();

		for(int i = 0; i < total_points; i++) {
			if(points[i].getID() == id_point) {
				points.erase(points.begin() + i);
				return true;
			}
		}
		return false;
	}

	double getCentralValue(int index) {
		return central_values[index];
	}

	void setCentralValue(int index, double value) {
		central_values[index] = value;
	}

	Point getPoint(int index) {
		return points[index];
	}

	int getTotalPoints() {
		return points.size();
	}

	int getID() {
		return id_cluster;
	}
};

class KMeans {
private: 
	int K;
	int total_values, total_points, max_iterations;
	vector<Cluster> clusters;

	int getIDNearestCenter(Point point) {
		double sum = 0.0; 
        double min_dist;
		int id_cluster_center = 0;

		for(int i = 0; i < total_values; i++)
			sum += pow(clusters[0].getCentralValue(i) - point.getValue(i), 2.0);

		min_dist = sqrt(sum);

		for(int i = 1; i < K; i++) {
			double dist;
			sum = 0.0;

			for(int j = 0; j < total_values; j++)
				sum += pow(clusters[i].getCentralValue(j) - point.getValue(j), 2.0);

			dist = sqrt(sum);
			if(dist < min_dist) {
				min_dist = dist;
				id_cluster_center = i;
			}
		}

		return id_cluster_center;
	}

public:
	KMeans(int K, int total_points, int total_values, int max_iterations) {
		this->K = K;
		this->total_points = total_points;
		this->total_values = total_values;
		this->max_iterations = max_iterations;
	}

    void print() {
		for(int i = 0; i < K; i++) {
			int total_points_cluster = clusters[i].getTotalPoints();
			cout << "Cluster " << clusters[i].getID() + 1 << endl;
			for(int j = 0; j < total_points_cluster; j++) {
				cout << "Point " << clusters[i].getPoint(j).getID() + 1 << ": ";
				for(int p = 0; p < total_values; p++)
					cout << clusters[i].getPoint(j).getValue(p) << " ";

				string point_name = clusters[i].getPoint(j).getName();

				if(point_name != "")
					cout << "- " << point_name;

				cout << endl;
			}
			cout << "Cluster values: ";
			for(int j = 0; j < total_values; j++)
				cout << clusters[i].getCentralValue(j) << " ";
            cout << "\n\n";
		}
    }

    // K-means|| 
    void init_kmeans_or(vector<Point> & points) {

        auto phase1 = chrono::high_resolution_clock::now();

        int oversampling_factor = 10;
        vector<Point> C;
        vector<Point> X = points;
        int rand_index = rand() % total_points;
        Point first_point = points[rand_index];
        C.push_back(first_point);

        double psi_double = 0;
        for (Point point : points) {
            psi_double += first_point.distance(point);
        }
        int psi_int = (int) log(psi_double);

        auto phase2 = chrono::high_resolution_clock::now();

        // generate random array 
        vector<double> rand_vec(X.size());
        for (int i = 0; i < X.size(); ++i) {
            double rand_num = ((double) rand() / (RAND_MAX));
            rand_vec[i] = rand_num;
        }

        double phase2A_total = 0;
        double phase2B_total = 0;
        for (int i = 0; i < psi_int; ++i) {
            int psi_arr[X.size()];

            auto phase2A = chrono::high_resolution_clock::now();

            tbb::parallel_for(
                tbb::blocked_range<int>(0, X.size()), 
                [&, i](tbb::blocked_range<int> r) {
                    for (int j = r.begin(); j != r.end(); ++j) {
                        double min = numeric_limits<double>::max();
                        for (int z = 0; z < C.size(); ++z) {
                            double d = X[j].distance(C[z]);
                            if (min > d * d) min = d * d;
                        }
                        psi_arr[j] = min;
                    }
                });

            double phi_c = 0;
            for (int j = 0; j < X.size(); ++j) phi_c += psi_arr[j];
            
            auto phase2B = chrono::high_resolution_clock::now();

            for (int j = 0; j < X.size(); ++j) {
                double prob_x = oversampling_factor * psi_arr[j] / phi_c;
                double rand_num = rand_vec[j];
                if (prob_x >= rand_num) {
                    C.push_back(X[j]);
                    X.erase(X.begin() + j);
                }
            }

            auto phase2C = chrono::high_resolution_clock::now();
            phase2A_total += (double) chrono::duration_cast<std::chrono::microseconds>(phase2B - phase2A).count();
            phase2B_total += (double) chrono::duration_cast<std::chrono::microseconds>(phase2C - phase2B).count();
        }

        auto phase3 = chrono::high_resolution_clock::now();

        tbb::concurrent_vector<int> min_indices;
        vector<int> weights(C.size());
        tbb::parallel_for(
            tbb::blocked_range<int>(0, X.size()), 
            [&](tbb::blocked_range<int> r) {
                for (int i = r.begin(); i != r.end(); ++i) {
                    double min = X[i].distance(C[0]);
                    int min_index = 0;
                    for (int j = 1; j < C.size(); j++) {
                        double d = X[i].distance(C[j]);
                        if (min > d) {
                            min = d;
                            min_index = j;
                        }
                    }
                    min_indices.push_back(min_index);
                }
        });

        for (auto min_index : min_indices) {
            weights[min_index] += 1;
        }

        
        auto phase4 = chrono::high_resolution_clock::now();

        for (int i = 0; i < C.size(); ++i) {
            weights[i] = weights[i] * weights[i];
        }

        vector<int> accu_weights(C.size()); 
        for (int i = 0; i < C.size(); ++i) {
            accu_weights[i] += weights[i];
            accu_weights[i] += i > 0 ? accu_weights[i - 1] : 0;
        }

        auto phase5 = chrono::high_resolution_clock::now();

        unordered_set<int> prohibited_indices;
        for (int i = 0; i < K; i++) {
			while(true) {
                int weight_total = accu_weights.back();
                int index_rand = rand() % weight_total;
                auto it = upper_bound(accu_weights.begin(), accu_weights.end(), index_rand);
                int index_point = (int) (it - accu_weights.begin());

                if (prohibited_indices.find(index_point) == prohibited_indices.end()) {
                    prohibited_indices.insert(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
        }

        auto phase6 = chrono::high_resolution_clock::now();

        cout << "Initialization Phase 1: " << chrono::duration_cast<std::chrono::microseconds>(phase2 - phase1).count() << endl;
        cout << "Initialization Phase 2: " << chrono::duration_cast<std::chrono::microseconds>(phase3 - phase2).count() << endl;
        cout << "Initialization Phase 2A: " << phase2A_total << endl;
        cout << "Initialization Phase 2B: " << phase2B_total << endl;
        cout << "Initialization Phase 3: " << chrono::duration_cast<std::chrono::microseconds>(phase4 - phase3).count() << endl;
        cout << "Initialization Phase 4: " << chrono::duration_cast<std::chrono::microseconds>(phase5 - phase4).count() << endl;
        cout << "Initialization Phase 5: " << chrono::duration_cast<std::chrono::microseconds>(phase6 - phase5).count() << endl;
    }

    void init_kmeans_plusplus(vector<Point> & points) {
        vector<Point> centers;
        unordered_set<int> prohibited_indices;
        int first_center_index = rand() % this->total_points;
        prohibited_indices.insert(first_center_index);
        points[first_center_index].setCluster(0);
        centers.push_back(points[first_center_index]);
        Cluster cluster(0, points[first_center_index]);
        this->clusters.push_back(cluster);

        for (size_t i = 1; i < this->K; ++i) {
            vector<double> weights(this->total_points);
            for (size_t j = 0; j < this->total_points; ++j) {
                double min_dist = numeric_limits<double>::max();
                for (Point center : centers) {
                    double d = points[j].distance(center); 
                    min_dist = d < min_dist ? d : min_dist;
                }
                weights[j] = min_dist * min_dist;
            }

            vector<double> accu_weights(this->total_points); 
            for (size_t i = 0; i < this->total_points; ++i) {
                accu_weights[i] += weights[i];
                accu_weights[i] += i > 0 ? accu_weights[i - 1] : 0;
            }

			while(true) {
                int weight_total = accu_weights.back();
                int index_rand = rand() % weight_total;
                auto it = upper_bound(accu_weights.begin(), accu_weights.end(), index_rand);
                int index_point = (int) (it - accu_weights.begin());
                if (prohibited_indices.find(index_point) == prohibited_indices.end()) {
                    prohibited_indices.insert(index_point);
                    centers.push_back(points[index_point]);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
        }
    }

    void init_kmeans_random(vector<Point> & points) {
		vector<int> prohibited_indexes;
		for (size_t i = 0; i < K; i++) {
			while (true) {
				int index_point = rand() % total_points;
				if (find(prohibited_indexes.begin(), prohibited_indexes.end(), index_point) == prohibited_indexes.end()) {
					prohibited_indexes.push_back(index_point);
					points[index_point].setCluster(i);
					Cluster cluster(i, points[index_point]);
					clusters.push_back(cluster);
					break;
				}
			}
		}
    }

	vector<double> run(vector<Point> & points, int init_method) {
        auto begin = chrono::high_resolution_clock::now();
        
		if (K > total_points) throw exception();

        if (init_method == 1)
            this->init_kmeans_random(points);
        else if (init_method == 2)
            this->init_kmeans_plusplus(points);
        else if (init_method == 3)
            this->init_kmeans_or(points);
        else
            throw exception();
        
        auto end_phase1 = chrono::high_resolution_clock::now();

		int iter = 1;

		while(true) {
			bool done = true;

            const double ERROR_THRESHHOLD = 0.01;
            typedef pair<int, int> task_t;
            vector<tbb::concurrent_vector<task_t>> cluster_task_vec(clusters.size());
            tbb::parallel_for(
                    tbb::blocked_range<int>(0, total_points), 
                    [&](tbb::blocked_range<int> r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            int id_old_cluster = points[i].getCluster();
                            int id_nearest_cluster = getIDNearestCenter(points[i]);

                            if (id_old_cluster != id_nearest_cluster) {
                                if (id_old_cluster != -1) {
                                    double olddist = clusters[id_old_cluster].distance(points[i]);
                                    double newdist = clusters[id_nearest_cluster].distance(points[i]);
                                    double error = olddist - newdist;
                                    if (error < ERROR_THRESHHOLD) continue;
                                    task_t remove_task = make_pair(points[i].getID(), -1);
                                    cluster_task_vec[id_old_cluster].push_back(remove_task);
                                }
                                points[i].setCluster(id_nearest_cluster);
                                task_t add_task = make_pair(points[i].getID(), 1);
                                cluster_task_vec[id_nearest_cluster].push_back(add_task);
                                done = false;
                            }
                        }
			        });

            tbb::parallel_for(
                    tbb::blocked_range<int>(0, cluster_task_vec.size()), 
                    [&](tbb::blocked_range<int> r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            for (auto task : cluster_task_vec[i]) {
                                int pid = task.first;
                                if (task.second == 1) clusters[i].addPoint(points[pid]);
                                else if (task.second == -1) clusters[i].removePoint(pid);
                            } 
                        }
			        });

            tbb::parallel_for(
                    tbb::blocked_range<int>(0, K), 
                    [&](tbb::blocked_range<int> r) {
                        for (size_t i = r.begin(); i != r.end(); ++i) {
                            for (size_t j = 0; j < total_values; j++){
                                int total_points_cluster = clusters[i].getTotalPoints();
                                double sum = 0.0;
                                if (total_points_cluster > 0) {
                                    for (size_t p = 0; p < total_points_cluster; p++)
                                        sum += clusters[i].getPoint(p).getValue(j);
                                    clusters[i].setCentralValue(j, sum / total_points_cluster);
                                }
                            }
                        }
			        });

			if(done == true || iter >= max_iterations) break;

			iter++;
		}

        auto end = chrono::high_resolution_clock::now();

        double total_exec = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
        double phase1_exec = std::chrono::duration_cast<std::chrono::microseconds>(end_phase1 - begin).count();
        double phase2_exec = std::chrono::duration_cast<std::chrono::microseconds>(end - end_phase1).count();
        vector<double> results{ total_exec, phase1_exec, phase2_exec, (double) iter };
        return results;
	}
};

void benchmark(int K, int total_points, int total_values, int max_iterations, vector<Point> points, int num_runs, int init_method) {
    double total_exec, phase1_exec, phase2_exec, iters = 0.0; 
    for (size_t i = 0; i < num_runs; ++i) 
    {
        KMeans kmeans_test(K, total_points, total_values, max_iterations);
        vector<double> results = kmeans_test.run(points, init_method);
        total_exec += results[0];
        phase1_exec += results[1];
        phase2_exec += results[2];
        iters += results[3];
    }

    double avg_total_exec = total_exec / num_runs;
    double avg_phase1_exec = phase1_exec / num_runs;
    double avg_phase2_exec = phase2_exec / num_runs;
    double avg_iters = iters / num_runs;
    cout << "\nAverage Results of " << num_runs << " runs and initialization method " << init_method << ": " << endl;
    cout << "Average Total Execution Time: " << avg_total_exec << endl;
    cout << "Average Iterations to Converge: " << avg_iters << endl;
    cout << "Average Phase 1 Execution Time: " << avg_phase1_exec << endl;
    cout << "Average Phase 2 Execution Time: " << avg_phase2_exec << endl;
}

int main(int argc, char *argv[]) {
	srand (time(NULL));

	int total_points, total_values, K, max_iterations, has_name;

	cin >> total_points >> total_values >> K >> max_iterations >> has_name;

	vector<Point> points;
	string point_name;

	for(size_t i = 0; i < total_points; i++) {
		vector<double> values;

		for(size_t j = 0; j < total_values; j++) {
			double value;
			cin >> value;
			values.push_back(value);
		}

		if(has_name) {
			cin >> point_name;
			Point p(i, values, point_name);
			points.push_back(p);
		} else {
			Point p(i, values);
			points.push_back(p);
		}
	}

    benchmark(K, total_points, total_values, max_iterations, points, 5, 3);
	return 0;
}
