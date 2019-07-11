def decrypt_command_line(sys_argv):
        command = {}

        for i in range(len(sys_argv)):
                if(sys_argv[i][0]=='-' and sys_argv[i][1]=='-' and i+1<len(sys_argv)):
                        command[sys_argv[i]] = sys_argv[i+1]

        return command