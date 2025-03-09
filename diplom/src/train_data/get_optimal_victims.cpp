#include <iostream>
#include <fstream>

#include <algorithm>
#include <vector>
#include <string>
#include <string_view>
#include <ranges>
#include <list>

constexpr size_t POSSIBLE_VICTIMS = 5;
using victims_t = std::list<std::pair<size_t, size_t>>;

// using Page = std::string;
struct Page
{
    size_t relId;
    size_t forkNum;
    size_t blockNum;

    bool operator==(const Page& rhs) const = default;
};

std::istream &operator>>(std::istream &is, Page& page)
{
    is >> page.relId >> page.forkNum >> page.blockNum;
    return is;
}

std::vector<Page> GetPages(std::string_view filename)
{
    std::vector<Page> res;
    std::ifstream file(filename.data());

    Page page;
    while( file >> page )
    {
        res.push_back(page);
    }

    return res;
}

size_t GetDistanceToNextAcc(const std::vector<Page>& pages, const Page& page, size_t currentIndex, size_t maxRate)
{
    size_t rate = 0;
    for (size_t i = currentIndex + 1; i < pages.size() && rate <= maxRate; ++i)
    {
        rate++;
        if (pages[i] == page)
            break;
    }

    return rate;
}

void UpdateRates(std::vector<size_t>& rates, const std::vector<Page>& pages, const std::vector<Page>& buffer, size_t currentIndex)
{
    for (size_t i = 0; i < buffer.size(); ++i)
    {
        if (rates[i] != 0)
            rates[i] -= 1;
    }

    for (size_t i = 0; i < buffer.size(); ++i)
    {
        if (rates[i] == 0)
        {
            // optimization by max rate
            bool isLastZero = std::any_of(rates.begin() + i + 1, rates.begin() + buffer.size(), [](size_t el) {return el == 0;});
            size_t maxRate = isLastZero ? std::ranges::max(rates) : pages.size();
            rates[i] = GetDistanceToNextAcc(pages, buffer[i], currentIndex, maxRate);
        }
    }
}

std::vector<victims_t> GetVictims(const std::vector<Page>& pages, size_t bufferSize)
{
    std::vector<victims_t> victims;
    std::vector<Page> buffer;
    std::vector<size_t> rates(bufferSize, 0);

    for (size_t i = 0; i < pages.size(); ++i)
    {
        const Page& page = pages[i];

        if (i % 10000 == 0)
            std::cout << i << std::endl;

        if (std::ranges::find(buffer, page) != buffer.end())
        {
            // already in buffer
            victims.push_back({});
        }
        else if (buffer.size() < bufferSize)
        {
            victims.push_back({{buffer.size(), 1}});
            buffer.push_back(page);
        }
        else
        {
            auto maxRateIt = std::ranges::max_element(rates);
            size_t victim = std::ranges::distance(rates.begin(), maxRateIt);
            buffer[victim] = page;
            victims.push_back({{victim, 1}});
            rates[victim] = 0;
            // std::cout << "victim: " << victim << std::endl;
        }

        UpdateRates(rates, pages, buffer, i);
    }

    return victims;
}

void SaveVictims(const std::vector<victims_t>& victims, std::string_view filename)
{
    std::ofstream file(filename.data());
    for (victims_t victims_rates : victims)
    {
        file << victims_rates.size() << std::endl;
        for (const auto& victim_rate : victims_rates)
            file << victim_rate.first << " " << victim_rate.second << std::endl;
    }
}

int main(int argc, char** argv)
{
    if (argc != 4)
    {
        std::cout << "Excpect arguments: <buf_size> <input file> <res file>" << std::endl;
        return 1;
    }
    std::cout << std::string(argv[1]) << std::endl;

    size_t bufSize = std::stoull(argv[1]);
    std::string inputFile(argv[2]);
    std::string outputFile(argv[3]);

    std::vector<Page> pages = GetPages(inputFile);

    std::vector<victims_t> victims = GetVictims(pages, bufSize);
    std::cout << victims.size() << std::endl;
    SaveVictims(victims, outputFile);

    return 0;
}