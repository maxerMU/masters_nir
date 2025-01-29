#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <string_view>
#include <ranges>
#include <list>

constexpr std::string_view INPUT_FILE_NAME = "acc_logfile2_pages";
constexpr std::string_view RESULT_FILE_NAME = "acc_logfile2_victims";
constexpr size_t BUF_SIZE = 32;
constexpr size_t POSSIBLE_VICTIMS = 5;
using victims_t = std::list<std::pair<size_t, size_t>>;

std::vector<size_t> GetPages(std::string_view filename)
{
    std::vector<size_t> res;
    size_t buf;
    std::ifstream file(filename.data());

    while( file >> buf )
    {
        res.push_back(buf);
    }

    return res;
}

victims_t FindOptimalVictims(const std::vector<size_t>& pages, std::vector<size_t>& buffer, size_t current_index)
{
    if (current_index == pages.size() - 1)
        return {};

    victims_t victims;

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        size_t rate = 0;
        for (size_t j = current_index; j < pages.size(); ++j)
        {
            rate++;
            if (buffer[i] == pages[j])
                break;
        }

        {
            auto it = victims.begin();
            while (it != victims.end() && it->second >= rate)
            {
                ++it;
            }

            if (it == victims.end() && victims.size() < POSSIBLE_VICTIMS)
            {
                victims.push_back({i, rate});
            }
            else if (it != victims.end())
            {
                victims.insert(it, {i, rate});
                if (victims.size() > POSSIBLE_VICTIMS)
                    victims.pop_back();
            }
            
        }
    }

    return victims;
}

std::vector<victims_t> GetVictims(const std::vector<size_t>& pages)
{
    std::vector<victims_t> victims;
    std::vector<size_t> buffer;

    for (size_t i = 0; i < pages.size(); ++i)
    {
        if (i % 10000 == 0)
            std::cout << i << std::endl;
        size_t page = pages[i];
        if (std::ranges::find(buffer, page) != buffer.end())
        {
            // already in buffer
            victims.push_back({});
        }
        else if (buffer.size() < BUF_SIZE)
        {
            victims.push_back({{buffer.size(), 1}});
            buffer.push_back(page);
        }
        else
        {
            victims_t victims_rates = FindOptimalVictims(pages, buffer, i);
            buffer[victims_rates.front().first] = page;
            victims.push_back(victims_rates);
            // std::cout << "victim: " << victim << std::endl;
        }
    }

    return victims;
}

void SaveVictims(const std::vector<victims_t>& victims, std::string_view filename)
{
    std::ofstream file(filename.data());
    for (victims_t victims_rates : victims)
    {
        file << victims_rates.size() << std::endl;
        for (const auto victim_rate : victims_rates)
            file << victim_rate.first << " " << victim_rate.second << std::endl;
    }
}

int main(void)
{
    std::vector<size_t> input = GetPages(INPUT_FILE_NAME);
    std::cout << input.size() << std::endl;

    // input.resize(1000);
    std::cout << input.size() << std::endl;

    std::vector<victims_t> victims = GetVictims(input);
    std::cout << victims.size() << std::endl;
    SaveVictims(victims, RESULT_FILE_NAME);

    return 0;
}