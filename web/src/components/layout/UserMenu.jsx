import { LogOut } from "lucide-react";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { useAuth } from "@/providers/contexts";

function initials(email) {
  return (email || "?").slice(0, 2).toUpperCase();
}

/** Signed-in account with a sign-out action. `showEmail` adds the address next to the avatar. */
export function UserMenu({ showEmail = false }) {
  const { user, signOut } = useAuth();
  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button
          variant="ghost"
          className={showEmail ? "h-auto w-full justify-start gap-3 px-2 py-2" : "size-9 rounded-full p-0"}
          aria-label={`Account: ${user?.email}`}
        >
          <Avatar className="size-8">
            <AvatarFallback className="bg-accent text-xs font-semibold text-accent-foreground">
              {initials(user?.email)}
            </AvatarFallback>
          </Avatar>
          {showEmail && (
            <span className="min-w-0 text-left">
              <span className="block truncate text-sm font-medium">{user?.email}</span>
              <span className="block text-xs text-muted-foreground">Signed in</span>
            </span>
          )}
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="end" className="w-56">
        <DropdownMenuLabel className="truncate font-normal text-muted-foreground">{user?.email}</DropdownMenuLabel>
        <DropdownMenuSeparator />
        <DropdownMenuItem onSelect={signOut}>
          <LogOut aria-hidden="true" /> Sign out
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
