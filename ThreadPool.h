#ifndef THREAD_POOL_H
#define THREAD_POOL_H

#include <vector>
#include <queue>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>
#include <functional>
#include <stdexcept>
#include "classification.hpp"

class ThreadPool {
public:
    ThreadPool(size_t, const std::string&, const std::string&, const std::string&);
    auto enqueue(std::string)-> std::future<std::string>;
    std::string getres(std::string);
    ~ThreadPool();
private:
    std::vector< std::thread > workers;
    std::vector<Classifier> classes;
    std::queue< std::shared_ptr< std::packaged_task<std::string(Classifier*)> > > tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;
};

inline ThreadPool::ThreadPool(size_t threads, const std::string& model_file, const std::string& trained_file, const std::string& mean_file)
    :   stop(false)
{
    for(size_t i = 0;i<threads;++i) {
        classes.emplace_back(model_file, trained_file, mean_file);
        workers.emplace_back(
                [this, i] {
                    for (; ;) {
                        std::shared_ptr< std::packaged_task<std::string(Classifier*)> > task;
                        {
                            std::unique_lock<std::mutex> lock(this->queue_mutex);
                            this->condition.wait(lock,
                                                 [this] { return this->stop || !this->tasks.empty(); });
                            if (this->stop && this->tasks.empty())
                                return;
                            task = this->tasks.front();
                            this->tasks.pop();
                        }
                        (*task)(&classes[i]);
                    }
                }
        );
    }
}

std::string dothejob(Classifier* w, std::string path){
    return w->Rec(path);
}


auto ThreadPool::enqueue(std::string f)
    -> std::future<std::string>
{
    auto task = std::make_shared< std::packaged_task<std::string(Classifier*)> >(
    std::bind(dothejob, std::placeholders::_1, std::forward<std::string>(f))
        );
        
    std::future<std::string> res = task->get_future();
    {
        std::unique_lock<std::mutex> lock(queue_mutex);

        if(stop)
            throw std::runtime_error("enqueue on stopped ThreadPool");
        tasks.push(task);
    }
    condition.notify_one();
    return res;
}

inline ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(queue_mutex);
        stop = true;
    }
    condition.notify_all();
    for(std::thread &worker: workers)
        worker.join();
}

inline std::string ThreadPool::getres(std::string path) {
    std::future<std::string> result;
    result = enqueue(path);
    return result.get();
}

#endif
