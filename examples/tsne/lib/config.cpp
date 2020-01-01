#include "config.h"
#include <fstream>

Config Config::load(const std::string& path) {
    std::ifstream infile(path);

    std::string key;
    Config config;

    while (infile >> key) {
        if (key == "n") {
            infile >> config.n;
        }
        else if (key == "input_dims") {
            infile >> config.input_dims;
        }
        else if (key == "output_dims") {
            infile >> config.output_dims;
        }
        else if (key == "theta") {
            infile >> config.theta;
        }
        else if (key == "perplexity") {
            infile >> config.perplexity;
        }
        else if (key == "max_iter") {
            infile >> config.max_iter;
        }
        else if (key == "use_ee") {
            infile >> config.use_ee;
        }
        else if (key == "ee_factor") {
            infile >> config.ee_factor;
        }
        else if (key == "ee_iter") {
            infile >> config.ee_iter;
        }
        else if (key == "use_periodic") {
            infile >> config.use_periodic;
        }
        else if (key == "periodic_cycle") {
            infile >> config.periodic_cycle;
        }
        else if (key == "periodic_duration") {
            infile >> config.periodic_duration;
        }
        else if (key == "periodic_reset_momentum") {
            infile >> config.periodic_reset_momentum;
        }
        else if (key == "input_path") {
            infile >> config.input_path;
        }
        else if (key == "output_path") {
            infile >> config.output_path;
        }
        else if (key == "eta") {
            infile >> config.eta;
        }
        else if (key == "momentum") {
            infile >> config.momentum;
        }
        else if (key == "log_per") {
            infile >> config.log_per;
        }
        else if (key == "ops") {
            infile >> config.ops;
        }
        else if (key == "cores") {
            infile >> config.cores;
        }
        else if (key == "add_point_weight") {
            infile >> config.add_point_weight;
        }
        else if (key == "update_index_weight") {
            infile >> config.update_index_weight;
        }
        else if (key == "tree_weight") {
            infile >> config.tree_weight;
        }
        else if (key == "table_weight") {
            infile >> config.table_weight;
        }
        else {
            throw std::runtime_error("unknown key: " + key);
        }
    }
    infile.close();

    // validate config
    
    if (config.n == 0 || config.input_dims == 0 || config.output_dims == 0 || config.input_path.size() == 0 || config.output_path.size() == 0) {
        throw; // std::exception("some of the required fields are missing: check n, input_dims, output_dims, input_path, output_path");
    }
   
    std::ifstream datafile(config.input_path);
    config.data.resize(config.n * config.input_dims);
    
    for (size_t i = 0; i < config.n; i++) {
        for (size_t j = 0; j < config.input_dims; j++) {
            datafile >> config.data[i * config.input_dims + j];
        }
    }

    datafile.close();

    config.event_file = std::make_shared<std::ofstream>(config.output_path + EVENT_LOG_PATH);

    return config;
}

void Config::save(double *Y) {
    std::ofstream outfile(output_path + RESULT_PATH);

    outfile << "n " << n << std::endl;
    outfile << "input_dims " << input_dims << std::endl;
    outfile << "output_dims " << output_dims << std::endl;
    outfile << "theta " << theta << std::endl;
    outfile << "perplexity " << perplexity << std::endl;
    outfile << "eta " << eta << std::endl;
    outfile << "momentum " << momentum << std::endl;
    outfile << "max_iter " << max_iter << std::endl;

    outfile << "use_ee " << use_ee << std::endl;
    outfile << "ee_factor " << ee_factor << std::endl;
    outfile << "ee_iter " << ee_iter << std::endl;
    outfile << "seed " << seed << std::endl;

    outfile << "use_periodic " << use_periodic << std::endl;
    outfile << "periodic_cycle " << periodic_cycle << std::endl;
    outfile << "periodic_duration " << periodic_duration << std::endl;

    outfile << "periodic_reset_momentum " << periodic_reset_momentum << std::endl;

    outfile << "input_path " << input_path << std::endl;
    outfile << "output_path " << output_path << std::endl;

    outfile << "log_per " << log_per << std::endl;

    outfile << "ops " << ops << std::endl;
    outfile << "cores " << cores << std::endl;

    outfile << "add_point_weight " << add_point_weight << std::endl;
    outfile << "update_index_weight " << update_index_weight << std::endl;
    outfile << "tree_weight " << tree_weight << std::endl;
    outfile << "table_weight " << table_weight << std::endl;

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < output_dims; j++) {
            outfile << Y[i * output_dims + j] << " ";
        }
        if(i + 1 < n) outfile << std::endl;
    }
    outfile.close();
}

void Config::save_embedding(size_t iter, double *Y) {
    std::ofstream outfile(output_path + EMBEDDING_PATH(iter));

    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < output_dims; j++) {
            outfile << Y[i * output_dims + j] << " ";
        }
        if (i + 1 < n) outfile << std::endl;
    }

    outfile.close();
}

void Config::event_log(const std::string& event_name, double time) {
    if (event_file && event_file->is_open()) {
        (*event_file) << event_name << " " << time << std::endl;
    }
}

void Config::event_log(const std::string& event_name, size_t iter, double error, double time) {
    if (event_file && event_file->is_open()) {
        (*event_file) << event_name << " " << iter << " " << error << " " << time << std::endl;
    }
}