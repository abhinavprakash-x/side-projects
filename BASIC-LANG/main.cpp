#include <iostream>
#include <vector>
#include <string>
#include <cstdint>

using namespace std;

bool strncmp(string s1, string s2, int n)
{
    if(s1.size() < n || s2.size() < n) return false;

    for(int i = 0; i < n; ++i)
    {
        if(s1[i] != s2[i]) return false;
    }
    return true;
}

struct symbolTable {
    string id;
    int val;
};

int main()
{
    vector<symbolTable> sym;
    string input;

    while(1)
    {
        cout << "$ ";
        cin >> input;

        if(input == "HELP")
        {
            cout << "Welcome to BASIC-LANG Help Menu.\n";
            cout << "BASIC-LANG is Written in C++ and Supports these: \n";
            cout << "HELP: Shows this Menu.\n"
                 << "EXIT: Stops Execution.\n"
                 << "PRINT <var>: Prints <var> on Console.\n"
                 << "SET <var> <val>: Creates a Variable and Assigns Value to it. \n";
        }

        else if(input == "EXIT") break; // Add Cleanup Later
        else if(strncmp(input, "PRINT ", 6))
        {
            int n = sym.size();

            int val = -1;
            string id = input.substr(6, 3);

            for(int i = 0; i < n; ++i)
            {
                if(sym[i].id == id)
                    val = sym[i].val;
            }

            if(val == -1) cout << "Unknown Variable. \n";
            else cout << val << "\n";
        }
        else if(strncmp(input, "SET ", 4))
        {
            string var_name = input.substr(4, 3);
            int val = 0; // temporarily

            symbolTable obj = {var_name, val};
            sym.push_back(obj);
        }
    }

    return 0;
}