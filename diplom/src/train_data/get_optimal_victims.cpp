#include <iostream>
#include <fstream>

#include <vector>
#include <string>
#include <string_view>
#include <ranges>

constexpr std::string_view INPUT_FILE_NAME = "acc_logfile2_pages";
constexpr std::string_view RESULT_FILE_NAME = "acc_logfile2_victims";
constexpr size_t BUF_SIZE = 32;

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

size_t FindOptimalVictim(const std::vector<size_t>& pages, std::vector<size_t>& buffer, size_t current_index)
{
    if (current_index == pages.size() - 1)
        return 0;

    size_t victim_rate = 0;
    size_t victim_ind = 0;

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        size_t rate = 0;
        for (size_t j = current_index; j < pages.size(); ++j)
        {
            rate++;
            if (buffer[i] == pages[j])
                break;
        }
        // std::cout << i << " " << rate << std::endl;
        if (rate > victim_rate)
        {
            victim_ind = i;
            victim_rate = rate;
        }
    }

    return victim_ind;
}

std::vector<size_t> GetVictims(const std::vector<size_t>& pages)
{
    std::vector<size_t> victims;
    std::vector<size_t> buffer;

    for (size_t i = 0; i < pages.size(); ++i)
    {
        if (i % 10000 == 0)
            std::cout << i << std::endl;
        size_t page = pages[i];
        if (std::ranges::find(buffer, page) != buffer.end())
        {
            // already in buffer
            victims.push_back(BUF_SIZE);
        }
        else if (buffer.size() < BUF_SIZE)
        {
            victims.push_back(buffer.size());
            buffer.push_back(page);
        }
        else
        {
            size_t victim = FindOptimalVictim(pages, buffer, i);
            buffer[victim] = page;
            victims.push_back(victim);
            // std::cout << "victim: " << victim << std::endl;
        }
    }

    return victims;
}

void SaveVictims(const std::vector<size_t>& victims, std::string_view filename)
{
    std::ofstream file(filename.data());
    for (size_t victim : victims)
        file << victim << std::endl;
}

int main(void)
{
    std::vector<size_t> input = GetPages(INPUT_FILE_NAME);
    std::cout << input.size() << std::endl;

    // input.resize(1000);
    std::cout << input.size() << std::endl;

    std::vector<size_t> victims = GetVictims(input);
    std::cout << victims.size() << std::endl;
    SaveVictims(victims, RESULT_FILE_NAME);

    return 0;
}