#pragma once
#include <fmt/format.h>
#include <stdexcept>
#define DG_HOST_ASSERT(cond, ...) do { if (!(cond)) throw std::runtime_error(std::string(#cond) __VA_OPT__(+ ": " + fmt::format(__VA_ARGS__))); } while (0)
#define DG_PANIC(...) throw std::runtime_error(fmt::format(__VA_ARGS__))
#define DG_STATIC_ASSERT(cond, ...) static_assert(cond, "" __VA_ARGS__)
