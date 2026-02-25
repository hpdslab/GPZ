#ifndef CPU_TIMER_HPP
#define CPU_TIMER_HPP

#include <chrono>
#include <string>
#include <iostream>
#include <iomanip>

class CpuTimer {
public:
    explicit CpuTimer(const std::string& name = "", bool auto_start = true)
            : name_(name), running_(false)
    {
        if (auto_start) start();
    }

    void start() {
        start_time_ = std::chrono::high_resolution_clock::now();
        running_ = true;
    }

    void stop() {
        end_time_ = std::chrono::high_resolution_clock::now();
        running_ = false;
    }

    double elapsed_seconds() const {
        return std::chrono::duration<double>(end_time_ - start_time_).count();
    }

    double elapsed_milliseconds() const {
        return std::chrono::duration<double, std::milli>(end_time_ - start_time_).count();
    }

    void print(const std::string& label = "") const {
        std::string tag = label.empty() ? name_ : label;
        if (!tag.empty()) tag += ": ";
        std::cout << std::fixed << std::setprecision(6)
                  << tag << elapsed_seconds() << " seconds" << std::endl;
    }

    ~CpuTimer() {
        if (running_) stop();
        if (!name_.empty())
            print();
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_, end_time_;
    std::string name_;
    bool running_;
};

#endif  // CPU_TIMER_HPP