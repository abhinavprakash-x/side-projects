#include <iostream>
#include <vector>
#include <string>
#include <sstream>

using namespace std;

struct symbolTable {
    string id;
    int val;
};

int main() {
    vector<symbolTable> sym;
    string input;

    while(1) {
        cout << "$ ";
        getline(cin, input);
        
        if(input.empty()) continue;

        istringstream iss(input);
        string command;
        iss >> command;

        if(command == "EXIT") break;

        else if(command == "HELP")
        {
            cout << "Welcome to BASIC-LANG Help Menu.\n";
            cout << "Commands:\n"
                 << "HELP: Shows this Menu.\n"
                 << "EXIT: Stops Execution.\n"
                 << "PRINT <var>: Prints <var> on Console.\n"
                 << "SET <var> <val>: Creates a Variable and Assigns Value to it.\n";
        }

        else if(command == "PRINT")
        {
            string var_name;
            iss >> var_name;

            if((var_name[0] == '\'' || var_name[0] == '\"') &&
                    (var_name[var_name.size() - 1] == '\'' || var_name[var_name.size() - 1] == '\"'))
            {
                cout << var_name.substr(1, var_name.size() - 2) << "\n";
                continue;
            }
            else if(var_name[0] == '\'' || var_name[0] == '\"')
            {
                cout << "Statements must begin and end with \" or \'." << "\n";
                continue;
            }

            int val = -1;
            for(const auto& item : sym)
            {
                if(item.id == var_name)
                {
                    val = item.val;
                    break;
                }
            }

            if(val == -1) cout << "Unknown Variable: " << var_name << ".\n";
            else cout << val << "\n";
        }

        else if(command == "SET")
        {
            string var_name;
            int val = 0;
            bool found = false;
            
            if (!(iss >> var_name)) continue; 
            iss >> val;

            if(var_name[0] == '\'' || var_name[0] == '\"')
            {
                cout << "Variables can't begin with \' or \"." << "\n";
                continue;
            }

            for(auto &item : sym)
            {
                if(item.id == var_name)
                {
                    found = true;
                    item.val = val;
                    break;
                }
            }
            if(!found) sym.push_back({var_name, val});
        }

        else cout << "Unknown command.\n";
    }

    return 0;
}