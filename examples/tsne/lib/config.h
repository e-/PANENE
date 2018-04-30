#ifndef panene_examples_tsne_config_h
#define panene_examples_tsne_config_h

#include <string>
#include <vector>
#include <memory>
#include <fstream>

#define EVENT_LOG_PATH "events.txt"
#define RESULT_PATH "result.txt"
#define EMBEDDING_PATH(iter) (std::to_string(iter) + ".txt")

class Config {
private:
    Config() = default;
    
public:    
    static Config load(const std::string& path);      
    Config(const Config &) = default;

    void save(double* Y);
    void event_log(const std::string& event_name, double time);
    void event_log(const std::string& event_name, size_t iter, double error, double time);
    void save_embedding(size_t, double*);

    size_t n = 0;
    size_t input_dims = 0;
    size_t output_dims = 0;
    double theta = 0.5;
    double perplexity = 10;
    double eta = 200;
    double momentum = .5;
    size_t max_iter = 1000;
    
    bool use_ee = true; // early exaggeration
    double ee_factor = 12.0;
    size_t ee_iter = 250;   
    int seed = -1;

    bool use_periodic = false;
    size_t periodic_cycle = 100;
    size_t periodic_duration = 30;
    
    bool periodic_reset_momentum = false;

    std::vector<double> data;
    
    std::string input_path;
    std::string output_path;

    std::shared_ptr<std::ofstream> event_file;
    size_t log_per = 10;

    size_t ops = 300;
    size_t cores = 4;

    size_t num_trees = 4;
    size_t num_checks = 1024;

    float add_point_weight = 0.7f;
    float update_index_weight = 0.3f;
    float tree_weight = 0.5f;
    float table_weight = 0.5f;
};

#endif
