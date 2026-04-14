#pragma once

#include <atomic>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <string>
#include <thread>

class AsyncMetricLogger
{
public:
    AsyncMetricLogger()
    {
        worker_ = std::thread(&AsyncMetricLogger::WorkerLoop, this);
    }

    ~AsyncMetricLogger()
    {
        Stop();
    }

    AsyncMetricLogger(const AsyncMetricLogger &) = delete;
    AsyncMetricLogger &operator=(const AsyncMetricLogger &) = delete;

    void Log(std::string line)
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            lines_.push(std::move(line));
        }
        cv_.notify_one();
    }

private:
    void Stop()
    {
        stop_requested_.store(true);
        cv_.notify_all();
        if (worker_.joinable())
        {
            worker_.join();
        }
    }

    void WorkerLoop()
    {
        while (true)
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait(lock, [this]()
                     { return stop_requested_.load() || !lines_.empty(); });
            if (lines_.empty())
            {
                if (stop_requested_.load())
                {
                    break;
                }
                continue;
            }
            std::string line = std::move(lines_.front());
            lines_.pop();
            lock.unlock();
            std::cout << line << std::endl;
        }
    }

    std::atomic<bool> stop_requested_{false};
    std::queue<std::string> lines_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::thread worker_;
};
