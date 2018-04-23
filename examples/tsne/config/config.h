#ifndef panene_examples_tsne_config_h
#define panene_examples_tsne_config_h

#include <string>
#include <vector>

class Config {
public:
    static Config load(const std::string& path);
    void save(double* Y);

    Config() = default;

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
};

#endif
